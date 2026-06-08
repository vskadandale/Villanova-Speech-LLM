import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import pandas as pd
import os
import argparse
    
VALID_LABELS = ['angry', 'happy', 'neutral', 'sad']

def load_data(file_path):
    """Loads SER results from a tab-separated file."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sample_id, label = parts
                data[sample_id] = label
    return data

def clean_prediction(raw_pred):
    """
    Post-processes generative output to map to valid categories.
    """
    raw_pred = raw_pred.strip().lower()
    
    # 1. Exact match check
    if raw_pred in VALID_LABELS:
        return raw_pred
    
    # 2. Substring check (e.g., if pred is "The emotion is happy")
    for label in VALID_LABELS:
        if label in raw_pred:
            return label
            
    # 3. Handle Junk: Return a string that is NOT in valid_labels 
    # but stays consistent for the metrics.
    return "invalid_junk"

def compute_ser_metrics(gt_file, pred_file):
    # Load ground truth and predictions
    gt_dict = load_data(gt_file)
    pred_dict = load_data(pred_file)
    
    
    # Align samples to ensure we compare the same IDs
    common_ids = sorted(list(set(gt_dict.keys()) & set(pred_dict.keys())))
    y_true = [gt_dict[sid].strip().lower() for sid in common_ids]
    y_pred = [clean_prediction(pred_dict[sid]) for sid in common_ids]
    
    # Weighted Accuracy (WA) is standard Accuracy
    wa = accuracy_score(y_true, y_pred)

    # Unweighted Accuracy (UWA) is the macro-average recall
    uwa = recall_score(y_true, y_pred, average='macro', labels=VALID_LABELS)
    
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    
    # Create a readable DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)

    return wa, uwa, cm_df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute SER accuracy from GT and prediction files.")
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth file path (required)")
    parser.add_argument("--pred_file", type=str, required=True, help="Predictions file path (required)")
    parser.add_argument("--cm_path", type=str, help="Path to save CM csv")

    args = parser.parse_args()

    gt_path = args.gt_file
    pred_path = args.pred_file

    wa, uwa, cm = compute_ser_metrics(gt_path, pred_path)
    
    # Save the Confusion Matrix if a path is provided
    if args.cm_path:
        cm.to_csv(args.cm_path)

    print(f"Weighted Accuracy (WA): {wa:.4f}")
    print(f"Unweighted Accuracy (UWA): {uwa:.4f}")
    print(f"RESULTS_SUMMARY {wa:.4f} {uwa:.4f}")
    print("\nConfusion Matrix:")
    print(cm)