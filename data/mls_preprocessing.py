import json
import os
import random
from pathlib import Path

# Set your base audio directory path here
DATASET_BASE_DIR = "/stek/corpora/mls_italian"
OUTPUT_BASE_DIR = "/stek/vskadandale/data/mls_italian_jsonl"
SEED = 42

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

for DATA_SUBSET in ["train", "dev", "test"]:
    AUDIO_BASE_DIR = os.path.join(DATASET_BASE_DIR, DATA_SUBSET, "audio")
    INPUT_FILE = os.path.join(DATASET_BASE_DIR, DATA_SUBSET, "transcripts.txt")
    OUTPUT_FILE = os.path.join(OUTPUT_BASE_DIR, f"{DATA_SUBSET}.jsonl")

    if not os.path.exists(INPUT_FILE):
        print(f"Skipping {DATA_SUBSET}: {INPUT_FILE} not found.")
        continue

    records = []
    missing_audio_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, target = parts

                # Parse speaker_id and book_id from key (e.g., '6001_10011_000000')
                key_parts = key.split("_")
                if len(key_parts) < 2:
                    continue

                speaker_id = key_parts[0]
                book_id = key_parts[1]

                # Reconstruct full path: AUDIO_BASE_DIR/6001/10011/6001_10011_000000.flac
                source_path = os.path.join(
                    AUDIO_BASE_DIR, speaker_id, book_id, f"{key}.flac"
                )

                if not os.path.exists(source_path):
                    missing_audio_count += 1
                    continue

                record = {
                    "key": f"mls_it_{key}",  # Prefixed to avoid key collisions in multi-dataset setups
                    "source": os.path.abspath(source_path),
                    "target": target.lower().strip(),
                }
                records.append(record)

    # --- SHUFFLE RECORDS ---
    random.seed(SEED)
    random.shuffle(records)

    # Write out to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for rec in records:
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[{DATA_SUBSET.upper()}] Successfully generated {OUTPUT_FILE} with"
        f" {len(records):,} shuffled records."
    )
    if missing_audio_count > 0:
        print(f"  Warning: {missing_audio_count} audio files were missing.")