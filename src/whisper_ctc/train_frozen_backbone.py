import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from dataclasses import dataclass
from typing import List
from omegaconf import OmegaConf
from transformers import TrainingArguments, Trainer, set_seed, TrainerCallback
from slam_llm.models.encoder import WhisperWrappedEncoder
import evaluate
import warnings

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

# 1. VOCAB
char_to_id = {
    "'": 27, "</s>": 2, "<pad>": 0, "<s>": 1, "<unk>": 3,
    "a": 7, "b": 24, "c": 19, "d": 14, "e": 5, "f": 20, "g": 21, "h": 11,
    "i": 10, "j": 29, "k": 26, "l": 15, "m": 17, "n": 9, "o": 8, "p": 23,
    "q": 30, "r": 13, "s": 12, "t": 6, "u": 16, "v": 25, "w": 18, "x": 28,
    "y": 22, "z": 31, "|": 4
}
id_to_char = {v: k for k, v in char_to_id.items()}
VOCAB_SIZE, BLANK_ID = len(char_to_id), 0

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f" Step: {state.global_step} | Loss: {logs['loss']:.4f}")

# 2. DATASET
class LocalJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, mel_size=128):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
        self.mel_size = mel_size

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = whisper.load_audio(item['source'])

        # Calculate length BEFORE padding
        # We store the actual (pre-padding) mel length for masking
        # If the original audio was 10s, mel_len is 1000
        raw_len = len(audio)
        actual_mel_len = min(3000, int(raw_len / 160))

        # SLAM-LLM internal logic: strictly 30s limit
        audio = whisper.pad_or_trim(audio)
        
        # This will now ALWAYS be 3000 frames
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size).permute(1, 0)
        
        labels = [char_to_id[c] for c in item.get('target', "").lower().replace(" ", "|") if c in char_to_id]
        return {"input_features": mel, "labels": labels, "mel_len": actual_mel_len}

@dataclass
class CTCDataCollator:
    def __call__(self, features):
        for f in features:
            f.pop("input_ids", None)

        # features is a list of dicts from your Dataset
        mels = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        mel_lens = torch.tensor([f["mel_len"] for f in features])

        # Pad Mel Spectrograms: (Channels, Time) -> (Time, Channels) for pad_sequence
        # Then permute back to (Batch, Channels, Time)
        mels_t = [m.permute(1, 0) for m in mels] 
        input_features_padded = torch.nn.utils.rnn.pad_sequence(
            mels_t, batch_first=True, padding_value=0.0
        ).permute(0, 2, 1)

        # Pad labels with -100 (ignored by most loss functions/metrics)
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_features": input_features_padded, 
            "labels": batch_labels, 
            "mel_lens": mel_lens,
        }
    
# 3. MODEL
class WhisperCTCModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder = WhisperWrappedEncoder.load(model_config)
        self.ctc_head = nn.Linear(model_config.encoder_dim, VOCAB_SIZE)
        
    def forward(self, input_features, labels=None, mel_lens=None, **kwargs):
        with torch.no_grad():
            hidden_states = self.encoder.extract_variable_length_features(input_features.permute(0, 2, 1))
        logits = self.ctc_head(hidden_states)
        
        # 2. Logit Masking (The "Pitfall" Fix)
        batch_size, t_out, vocab_size = logits.shape
        out_lens = (mel_lens + 1) // 2

        # Create mask: True for valid frames, False for padding
        mask = torch.arange(t_out, device=logits.device).expand(batch_size, t_out) < out_lens.unsqueeze(1)
        # Fill padding with very low values
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        
        if labels is not None and mel_lens is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1) # (T, B, V)
            
            # Whisper encoder downsamples 3000 frames -> 1500 frames
            #input_lengths = (mel_lens + 1) // 2 
            #input_lengths = torch.clamp((mel_lens + 1) // 2, max=logits.size(1))
            input_lengths = torch.clamp(out_lens, max=t_out)
            target_lengths = (labels != -100).sum(dim=-1)

            # Flatten labels for F.ctc_loss (remove -100 padding)
            targets = labels[labels != -100]

            loss = F.ctc_loss(
                log_probs, 
                targets, 
                input_lengths, 
                target_lengths, 
                blank=BLANK_ID, 
                zero_infinity=True # Keep this as a safety net
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# 4. METRICS & MAIN
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    logits = torch.from_numpy(pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions).float()
    pred_ids = torch.argmax(logits, dim=-1)
    
    def decode(ids, is_label=False):
        res, prev = [], None
        for i in ids:
            val = i.item()
            if is_label:
                if val != -100: res.append(id_to_char.get(val, ""))
            elif val != prev and val != BLANK_ID:
                res.append(id_to_char.get(val, ""))
            prev = val
        return "".join(res).replace("|", " ").strip()

    decoded_preds = [decode(ids) for ids in pred_ids]
    decoded_labels = [decode(lbl, True) for lbl in torch.from_numpy(pred.label_ids)]
    print(f"\n[EVAL] REF: {decoded_labels[0]} \n PRED: {decoded_preds[0]}")
    return {"wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

def main():
    set_seed(42)
    model = WhisperCTCModel(OmegaConf.create({
        "encoder_path": "large-v3-turbo", "encoder_dim": 1280,
        "encoder_name": "whisper", "whisper_decode": False, "encoder_path_hf": None,
    }))
    for p in model.encoder.parameters(): p.requires_grad = False

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./exp/whisper_ctc_frozen_backbone", per_device_train_batch_size=8,
            learning_rate=1e-4, num_train_epochs=3, fp16=True,
            eval_strategy="steps", eval_steps=1000, logging_steps=100,
            remove_unused_columns=False,
            save_steps=1000,
            save_strategy="steps",
            dataloader_num_workers=4,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            # --- NEW: GRADIENT CLIPPING ---
            max_grad_norm=1.0,
            # Best Model Tracking
            load_best_model_at_end=True,
            metric_for_best_model="wer", # This maps to the 'wer' key in compute_metrics
            greater_is_better=False,     # Lower WER is better
            save_total_limit=3,
        ),
        train_dataset=LocalJsonlDataset("/stek/vskadandale/librispeech_jsonl/train-libri.jsonl"),
        eval_dataset=LocalJsonlDataset("/stek/vskadandale/librispeech_jsonl/librispeech_dev_other.jsonl"),
        data_collator=CTCDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback()]
    )
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    torch.save(model.state_dict(), "./exp/whisper_ctc_frozen_backbone/whisper_ctc_frozen_backbone.pth")

if __name__ == "__main__":
    main()