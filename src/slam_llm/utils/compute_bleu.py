import sys
import os
import json
from evaluate import load
from whisper.normalizers import BasicTextNormalizer

def parse_slam_file(file_path):
    """Parses SLAM-LLM output files (key \t text)."""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return data
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                # Strip keys just in case there is trailing whitespace
                data[parts[0].strip()] = parts[1].strip()
    return data

def main():
    if len(sys.argv) != 4:
        print("Usage: python compute_bleu.py <gt_file> <pred_file> <report_path>")
        sys.exit(1)

    gt_path, pred_path, report_path = sys.argv[1:]

    # Load resources
    bleu_metric = load("bleu")
    normalizer = BasicTextNormalizer()

    # Match data
    gt_dict = parse_slam_file(gt_path)
    pred_dict = parse_slam_file(pred_path)
    
    common_keys = [k for k in gt_dict if k in pred_dict]
    
    if not common_keys:
        print(f"Error: No matching keys found between {gt_path} and {pred_path}")
        return

    # Prepare lists for evaluate
    preds = [pred_dict[k] for k in common_keys]
    refs = [[gt_dict[k]] for k in common_keys] # Wrapped for single-ref BLEU

    # 1. Raw BLEU
    raw_results = bleu_metric.compute(predictions=preds, references=refs)
    
    # 2. Normalized BLEU
    norm_preds = [normalizer(p) for p in preds]
    norm_refs = [[normalizer(r[0])] for r in refs]
    norm_results = bleu_metric.compute(predictions=norm_preds, references=norm_refs)

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation on {len(common_keys)} segments\n")
        f.write(f"Raw BLEU: {raw_results['bleu'] * 100:.2f}\n")
        f.write(f"Normalized BLEU: {norm_results['bleu'] * 100:.2f}\n")

if __name__ == "__main__":
    main()
