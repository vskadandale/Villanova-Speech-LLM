import csv
import json
import random
from pathlib import Path


def process_fleurs_split(
    data_dir: Path, split_name: str, output_jsonl: Path, seed: int = 42
) -> None:
    """Processes, shuffles, and converts a FLEURS split into SLAM-LLM JSONL format.

    Args:

    data_dir: Path to language root (e.g., /stek/corpora/fleurs_europe/data/it_it)
    split_name: Name of the split ('train', 'dev', or 'test') output_jsonl: Path
    where the resulting .jsonl file will be written seed: Random seed for
    reproducible shuffling
    """
    lang = data_dir.parts[-1]
    tsv_file = data_dir / f"{split_name}.tsv"
    audio_dir = data_dir / "audio" / split_name

    if not tsv_file.exists():
        print(f"Error: {tsv_file} not found. Skipping {split_name}.")
        return

    if not audio_dir.exists():
        print(f"Error: Audio directory {audio_dir} not found. Skipping.")
        return

    records = []
    missing_audio_count = 0

    with open(tsv_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split by TAB
            parts = line.split("\t")
            if len(parts) < 4:
                continue

            sample_id = parts[0].strip()
            file_name = parts[1].strip()
            # Column 3 contains normalized text, fall back to column 2 if empty
            raw_text = parts[3].strip() if parts[3].strip() else parts[2].strip()

            audio_path = audio_dir / file_name

            if not audio_path.exists():
                missing_audio_count += 1
                continue

            # Build a unique key combining split, sentence ID, and audio filename ID
            file_stem = Path(file_name).stem
            key = f"fleurs_{lang}_{split_name}_{sample_id}_{file_stem}"

            record = {
                "key": key,
                "source": str(audio_path.resolve()),
                "target": raw_text.lower().strip(),
            }
            records.append(record)

    # --- SHUFFLE DATA ---
    random.seed(seed)
    random.shuffle(records)

    # Write output to JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[{split_name.upper()}] Wrote {len(records)} shuffled records to"
        f" {output_jsonl}"
    )
    if missing_audio_count > 0:
        print(
            f"  Warning: {missing_audio_count} audio files missing in"
            f" {audio_dir}"
        )


def main():
    FLEURS_BASE_DIR = Path("/stek/corpora/fleurs_europe/data/")
    langs = ["it_it", "en_us"]
    
    splits = ["train", "dev", "test"]

    print("--- Processing & Shuffling FLEURS Italian ---")
    for lang in langs:
        output_base_dir = Path(f"/stek/vskadandale/data/fleurs_{lang}_jsonl")
        for split in splits:
            data_dir = FLEURS_BASE_DIR / lang 
            output_jsonl = output_base_dir / f"{split}.jsonl"
            process_fleurs_split(data_dir, split, output_jsonl)


if __name__ == "__main__":
    main()