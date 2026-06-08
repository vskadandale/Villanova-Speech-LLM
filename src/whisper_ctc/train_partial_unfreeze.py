import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import random
import evaluate
import warnings
from dataclasses import dataclass
from omegaconf import OmegaConf
from transformers import TrainingArguments, Trainer, set_seed, TrainerCallback
from slam_llm.models.encoder import WhisperWrappedEncoder

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

# 1. VOCAB & CONSTANTS
char_to_id = {
    "'": 27, "</s>": 2, "<pad>": 0, "<s>": 1, "<unk>": 3,
    "a": 7, "b": 24, "c": 19, "d": 14, "e": 5, "f": 20, "g": 21, "h": 11,
    "i": 10, "j": 29, "k": 26, "l": 15, "m": 17, "n": 9, "o": 8, "p": 23,
    "q": 30, "r": 13, "s": 12, "t": 6, "u": 16, "v": 25, "w": 18, "x": 28,
    "y": 22, "z": 31, "|": 4
}
id_to_char = {v: k for k, v in char_to_id.items()}
VOCAB_SIZE, BLANK_ID = len(char_to_id), 0

# 2. DATA & COLLATOR (With Adaptive SpecAugment)
class LocalJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, mel_size=128):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
        self.mel_size = mel_size
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        audio = whisper.load_audio(item['source'])
        raw_len = len(audio)
        actual_mel_len = min(3000, int(raw_len / 160))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size).permute(1, 0)
        labels = [char_to_id[c] for c in item.get('target', "").lower().replace(" ", "|") if c in char_to_id]
        return {"input_features": mel, "labels": labels, "mel_len": actual_mel_len}

@dataclass
class CTCDataCollator:
    model: nn.Module
    def __call__(self, features):
        mels = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        mel_lens = torch.tensor([f["mel_len"] for f in features])

        if self.model.training:
            mels = [self.spec_augment(m, mel_len=l.item()) for m, l in zip(mels, mel_lens)]

        mels_t = [m.permute(1, 0) for m in mels] 
        input_features_padded = torch.nn.utils.rnn.pad_sequence(mels_t, batch_first=True, padding_value=0.0).permute(0, 2, 1)
        batch_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features_padded, "labels": batch_labels, "mel_lens": mel_lens}

    def spec_augment(self, mel, n_freq_mask=2, f_width=20, n_time_mask=2, mel_len=None):
        mel = mel.clone()
        n_mels, n_steps = mel.shape
        actual_t = mel_len if mel_len is not None else n_steps
        for _ in range(n_freq_mask):
            f = random.randint(0, f_width); f0 = random.randint(0, n_mels - f); mel[f0:f0+f, :] = 0
        max_t_width = min(25, int(actual_t * 0.15)) 
        if max_t_width > 2:
            for _ in range(n_time_mask):
                t = random.randint(0, max_t_width); t0 = random.randint(0, actual_t - t); mel[:, t0:t0+t] = 0
        return mel

# 3. MODEL (With Entropy Regularization)
class WhisperCTCModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder = WhisperWrappedEncoder.load(model_config)
        self.ctc_head = nn.Linear(model_config.encoder_dim, VOCAB_SIZE)
        
    def forward(self, input_features, labels=None, mel_lens=None, **kwargs):
        # Allow gradients if we are unfreezing
        hidden_states = self.encoder.extract_variable_length_features(input_features.permute(0, 2, 1))
        logits = self.ctc_head(hidden_states)
        
        # Masking
        batch_size, t_out, _ = logits.shape
        out_lens = (mel_lens + 1) // 2
        mask = torch.arange(t_out, device=logits.device).expand(batch_size, t_out) < out_lens.unsqueeze(1)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        
        if labels is not None and mel_lens is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.clamp(out_lens, max=t_out)
            target_lengths = (labels != -100).sum(dim=-1)
            targets = labels[labels != -100]

            # A. CTC Loss
            loss_ctc = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK_ID, zero_infinity=True)

            # B. Entropy Regularization (Label Smoothing Effect)
            # We only calculate entropy for the valid (unmasked) parts
            probs = F.softmax(logits, dim=-1)
            p_log_p = (probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            entropy = -(p_log_p * mask).sum() / mask.sum() 
            
            beta = 0.05 # Smoothing strength
            total_loss = loss_ctc - (beta * entropy)
            
            return {"loss": total_loss, "logits": logits}
        return {"logits": logits}

# 4. METRICS & TRAINING
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
            if not is_label: prev = val
        return "".join(res).replace("|", " ").strip()
    decoded_preds = [decode(ids) for ids in pred_ids]
    decoded_labels = [decode(lbl, True) for lbl in torch.from_numpy(pred.label_ids)]
    print(f"\n REF: {decoded_labels[0]} \n PRED: {decoded_preds[0]}")
    return {"wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

def main():
    set_seed(42)
    model = WhisperCTCModel(OmegaConf.create({
        "encoder_path": "large-v3-turbo", "encoder_dim": 1280,
        "encoder_name": "whisper", "whisper_decode": False, "encoder_path_hf": None,
    }))

    # UNFREEZING STRATEGY: Freeze all, then unfreeze the last 4 Transformer blocks
    for p in model.encoder.parameters(): p.requires_grad = False
    for block in model.encoder.blocks[-4:]:
        for p in block.parameters(): p.requires_grad = True

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./exp/whisper_ctc_partial_unfreeze", per_device_train_batch_size=8,
            learning_rate=2e-5, num_train_epochs=3, fp16=True,
            eval_strategy="steps", eval_steps=1000, logging_steps=100, save_steps=1000,
            warmup_steps=500, weight_decay=0.01, load_best_model_at_end=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,  # CRITICAL FIX
            metric_for_best_model="wer", greater_is_better=False, save_total_limit=2,
        ),
        train_dataset=LocalJsonlDataset("/stek/vskadandale/librispeech_jsonl/train-libri.jsonl"),
        eval_dataset=LocalJsonlDataset("/stek/vskadandale/librispeech_jsonl/librispeech_dev_other.jsonl"),
        data_collator=CTCDataCollator(model=model),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    torch.save(model.state_dict(), "./exp/whisper_ctc_partial_unfreeze/whisper_ctc_partial_unfreeze.pth")

if __name__ == "__main__":
    main()