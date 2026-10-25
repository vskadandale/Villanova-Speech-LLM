import json
from pathlib import Path

# ITALIC: spoken Italian version of MASSIVE, already one 16 kHz mono wav per utterance
ITALIC_DIR = Path("/stek/corpora/italic/zenodo_dataset")
OUTPUT_JSONL_DIR = Path("/stek/vskadandale/data/italic_jsonl")


def process_split(config, split):
    """Converts one ITALIC split (e.g. massive_test.json) into SLAM-LLM JSONL format.

    Args:
        config: ITALIC configuration ('massive', 'hard_speaker' or 'hard_noisy')
        split: Name of the split ('train', 'validation' or 'test')
    """
    metadata_file = ITALIC_DIR / f"{config}_{split}.json"
    records = []
    missing_audio_count = 0

    # Metadata has one JSON object per line
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            audio_path = ITALIC_DIR / "recordings" / f"{sample['id']}.wav"
            if not audio_path.exists():
                missing_audio_count += 1
                continue

            records.append({
                "key": f"italic_{sample['id']}",
                "source": str(audio_path),
                # MASSIVE utterances are already lowercase with numbers spelled out
                "target": " ".join(sample["utt"].split()).lower(),
            })

    output_jsonl = OUTPUT_JSONL_DIR / f"{config}_{split}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[{config} {split}] Wrote {len(records)} records to {output_jsonl}")
    if missing_audio_count > 0:
        print(f"  Warning: {missing_audio_count} audio files missing")


if __name__ == "__main__":
    # 'massive' is the standard split; the 'hard_*' ones hold out unseen speakers or
    # noisier recording conditions
    for config in ["massive", "hard_speaker", "hard_noisy"]:
        process_split(config, "test")
