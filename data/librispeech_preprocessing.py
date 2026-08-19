import json
from pathlib import Path
import random


def process_librispeech(split_folder_path: str, output_path: str, seed: int = 42) -> None:
    """Converts a LibriSpeech dataset directory into SLAM-LLM JSONL format.

    Args:
        split_folder_path: Path to the LibriSpeech split directory (e.g.,
          /data/LibriSpeech/train-clean-100)
        output_path: Path for the output .jsonl file
    """
    base_dir = Path(split_folder_path)
    output_path = Path(output_path)

    records = []
    missing_audio_count = 0

    # Recursively find all *.trans.txt files
    trans_files = sorted(base_dir.rglob("*.trans.txt"))

    print(
        f"Found {len(trans_files)} transcription files in {base_dir.resolve()}..."
    )

    for trans_file in trans_files:
        trans_dir = trans_file.parent

        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # LibriSpeech format: <KEY> <TRANSCRIPT IN ALL CAPS>
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue

                key = parts[0]
                raw_text = parts[1]

                # Match key to .flac audio file
                audio_path = trans_dir / f"{key}.flac"

                if not audio_path.exists():
                    missing_audio_count += 1
                    continue

                # Build record with lowercased target transcript
                record = {
                    "key": key,
                    "source": str(audio_path.resolve()),
                    "target": raw_text.lower().strip(),
                }
                records.append(record)

    # --- SHUFFLE HERE BEFORE WRITING ---
    random.seed(seed)
    random.shuffle(records)

    # Write output to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Successfully created {output_path} with {len(records)} records."
    )
    if missing_audio_count > 0:
        print(f"Warning: {missing_audio_count} audio files were missing.")


if __name__ == "__main__":
    # Example usage: update paths as needed
    LIBRISPEECH_BASE_DIR = "/stek/corpora/LibriSpeech"
    OUTPUT_BASE_DIR = "/stek/vskadandale/data/librispeech_jsonl/"
    splits = ["test-clean",
              "test-other",
              "dev-clean",
              "dev-other",
              "train-clean-100",
              "train-clean-360",
              "train-other-500"]

    for split in splits:
        split_folder_path = Path(LIBRISPEECH_BASE_DIR) / split
        output_path = Path(OUTPUT_BASE_DIR) / f"{split}.jsonl"
        process_librispeech(split_folder_path, output_path)