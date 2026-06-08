#PARTIAL UNFREEZE RESULTS on LIBRISPEECH TEST CLEAN
#Final Test WER: 2.75%
#Average Blank Density: 49.49%
#FROZEN BACKBONE RESULTS on LIBRISPEECH TEST CLEAN
#Final Test WER: 36.59%
#Average Blank Density: 39.45%
import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from omegaconf import OmegaConf
import evaluate
from torch.utils.data import DataLoader

# Load standard WER metric
wer_metric = evaluate.load("wer")

# --- 1. MODEL & DATASET DEFINITIONS (Must match training) ---
from train_partial_unfreeze import WhisperCTCModel, LocalJsonlDataset, CTCDataCollator, id_to_char, BLANK_ID

@torch.no_grad()
def get_blank_stats(logits, mel_lens):
    # Get the most likely token for every frame
    pred_ids = torch.argmax(logits, dim=-1)
    batch_size, t_max = pred_ids.shape
    out_lens = (mel_lens + 1) // 2

    ratios = []
    for b in range(batch_size):
        # We only look at the valid frames for this specific audio length
        valid_frames = pred_ids[b, :out_lens[b]]
        num_blanks = (valid_frames == BLANK_ID).sum().item()
        ratios.append(num_blanks / out_lens[b].item())
    
    return ratios

def greedy_decode_batch(logits, mel_lens, id_to_char):
    batch_size, t_out, _ = logits.shape
    out_lens = (mel_lens + 1) // 2
    pred_ids = torch.argmax(logits, dim=-1)
    
    decoded_texts = []
    for b in range(batch_size):
        res, prev = [], None
        # Only iterate up to the actual length of this specific audio
        for i in range(out_lens[b]):
            val = pred_ids[b, i].item()
            if val != prev and val != BLANK_ID:
                res.append(id_to_char.get(val, ""))
            prev = val
        decoded_texts.append("".join(res).replace("|", " ").strip())
    return decoded_texts

# --- 2. THE TEST RUNNER ---
def run_test(checkpoint_path, test_jsonl, output_txt="whisper_ctc_inference_results.log"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Model
    config = OmegaConf.create({
        "encoder_path": "large-v3-turbo", "encoder_dim": 1280,
        "encoder_name": "whisper", "whisper_decode": False, "encoder_path_hf": None,
    })
    model = WhisperCTCModel(config)
    
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # Prepare Data
    dataset = LocalJsonlDataset(test_jsonl)
    loader = DataLoader(
        dataset, 
        batch_size=16, 
        num_workers=4, 
        collate_fn=CTCDataCollator(model=model)
    )

    all_preds = []
    all_refs = []
    all_blank_ratios = [] # To track density

    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_features = batch["input_features"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            labels = batch["labels"] # CPU is fine for labels

            # Forward
            logits = model(input_features, mel_lens=mel_lens)["logits"]
            
            # 1. Count Blanks
            batch_ratios = get_blank_stats(logits, mel_lens)
            all_blank_ratios.extend(batch_ratios)

            # Decode predictions
            preds = greedy_decode_batch(logits, mel_lens, id_to_char)
            all_preds.extend(preds)

            # Decode references (labels)
            for lbl in labels:
                actual = "".join([id_to_char[i.item()] for i in lbl if i.item() != -100])
                all_refs.append(actual.replace("|", " ").strip())

    # Calculate final WER
    final_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    avg_blank_density = sum(all_blank_ratios) / len(all_blank_ratios)

    # Save results to file for inspection
    with open(output_txt, "w") as f:
        for r, p in zip(all_refs, all_preds):
            f.write(f"REF: {r}\nPRED: {p}\n---\n")

    print(f"\nFinal Test WER: {final_wer * 100:.2f}%")
    print(f"Average Blank Density: {avg_blank_density * 100:.2f}%")
    print(f"Results saved to {output_txt}")

if __name__ == "__main__":
    run_test(
        checkpoint_path="./models/whisper_ctc_frozen_backbone.pth",
        test_jsonl="/stek/vskadandale/librispeech_jsonl/librispeech_test_clean.jsonl"
    )