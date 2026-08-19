import json
import os
import random
from pathlib import Path

# Base directory for the corpus
BASE_DIR = Path("/stek/corpora/common_voice_17_europe")

# Splits to process: "train", "validation", "test"
SPLITS = ["train", "validation", "test"]
TARGET_LANGS = ["Italian_it", "English_en"]
OUTPUT_BASE_DIR = Path("/stek/vskadandale/data/")

# Fixed seed for reproducible shuffling
SEED = 42


def process_subset(base_dir: Path, split_name: str, seed: int = SEED):
    split_dir = base_dir / split_name
    output_jsonl = output_dir / f"{split_name}.jsonl"

    if not split_dir.exists():
        print(f"Directory {split_dir} does not exist.")
        return

    # Find the matching directory (e.g., Italian_it_train)
    lang_dirs = [
        d
        for d in split_dir.iterdir()
        if d.is_dir() and d.name.startswith(TARGET_LANG_PREFIX)
    ]

    if not lang_dirs:
        print(
            f"No directory matching prefix '{TARGET_LANG_PREFIX}' found in"
            f" {split_dir}"
        )
        return

    lang_dir = lang_dirs[0]
    records = []

    # Match each .txt file with its corresponding .mp3 file
    for txt_path in lang_dir.glob("*.txt"):
        key = txt_path.stem  # e.g., '126885'
        mp3_path = lang_dir / f"{key}.mp3"

        # Skip if the corresponding audio file is missing
        if not mp3_path.exists():
            continue

        # Read the transcript from the text file
        with open(txt_path, "r", encoding="utf-8") as tf:
            target_text = tf.read().lower().strip()

        if not target_text:
            continue

        record = {
            "key": f"{lang}_{key}",  # Prefixed language to avoid key collisions across datasets
            "source": str(mp3_path.resolve()),
            "target": target_text,
        }
        records.append(record)

    # --- SHUFFLE RECORDS ---
    random.seed(seed)
    random.shuffle(records)

    # Write shuffled records to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[{split_name.upper()}] Written {len(records):,} shuffled {lang}"
        f" records to {output_jsonl.name}"
    )


if __name__ == "__main__":
    for lang in TARGET_LANGS:
        TARGET_LANG_PREFIX = f"{lang}_"
        output_dir = OUTPUT_BASE_DIR / f"cv17_{lang}_jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing language: {lang}")
        for split in SPLITS:
            process_subset(BASE_DIR, split)