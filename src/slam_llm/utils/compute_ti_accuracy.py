import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

VALID_LABELS = ['entertainment', 'sports', 'science/technology', 'politics', 'travel', 'geography', 'health']

def load_data(file_path):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--cm_path", type=str)
    args = parser.parse_args()

    gt_dict = load_data(args.gt_file)
    pred_dict = load_data(args.pred_file)

    # Use GT keys as the master list
    keys = sorted(gt_dict.keys())
    y_true = [gt_dict[sid].strip().lower() for sid in keys]
    y_pred = [clean_prediction(pred_dict.get(sid, "")) for sid in keys]


    # Compute Metrics
    wa = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=VALID_LABELS, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=VALID_LABELS, zero_division=0)
    uwa = recall_score(y_true, y_pred, average='macro', labels=VALID_LABELS, zero_division=0)

    # Output for Bash extraction
    print(f"METRIC_WA: {wa:.4f}")
    print(f"METRIC_F1M: {f1_macro:.4f}")
    print(f"METRIC_F1W: {f1_weighted:.4f}")
    print(f"METRIC_UWA: {uwa:.4f}")

    # Confusion Matrix
    if args.cm_path:
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(args.cm_path)

if __name__ == "__main__":
    main()