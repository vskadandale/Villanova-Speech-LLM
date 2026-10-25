import json
from pathlib import Path

import soundfile as sf

# The "legacy" distribution keeps the TED-LIUM 1/2 dev and test sets, which is the
# test set results are usually reported on (11 talks, 1155 segments)
TEDLIUM_DIR = Path("/stek/corpora/TEDLIUM_release-3/legacy")
OUTPUT_AUDIO_DIR = Path("/stek/corpora/tedlium_segmented")
OUTPUT_JSONL_DIR = Path("/stek/vskadandale/data/tedlium_jsonl")


def read_stm(stm_file):
    """Yields (start, end, text) for each scored segment of an STM file."""
    with open(stm_file, "r", encoding="utf-8") as f:
        for line in f:
            # STM format: <talk> <channel> <speaker> <start> <end> <label> <transcript>
            parts = line.split(maxsplit=6)
            if len(parts) < 7:
                continue

            # Silences between segments are marked as not to be scored
            text = parts[6]
            if "ignore_time_segment_in_scoring" in text:
                continue

            # Transcripts are lowercase with contractions split off ("i 'm", "won 't")
            text = " ".join(text.split()).replace(" '", "'")
            yield float(parts[3]), float(parts[4]), text


def process_split(split):
    print(f"--- Processing TED-LIUM 3 {split.upper()} ---")
    split_dir = TEDLIUM_DIR / split
    output_audio_dir = OUTPUT_AUDIO_DIR / split
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_duration = 0.0
    for stm_file in sorted((split_dir / "stm").glob("*.stm")):
        talk_id = stm_file.stem
        # SPH audio is already 16 kHz mono PCM, so cutting a segment is slicing samples
        audio, sr = sf.read(split_dir / "sph" / f"{talk_id}.sph", dtype="int16")

        for start, end, text in read_stm(stm_file):
            # Unique ID: TalkID_Start_End in ms (e.g. BillGates_2010_0015861_0019986)
            key = f"{talk_id}_{round(start * 1000):07d}_{round(end * 1000):07d}"
            output_path = output_audio_dir / f"{key}.wav"
            if not output_path.exists():
                segment = audio[round(start * sr):round(end * sr)]
                sf.write(output_path, segment, sr, subtype="PCM_16")

            records.append({"key": key, "source": str(output_path), "target": text})
            total_duration += end - start

    output_jsonl = OUTPUT_JSONL_DIR / f"{split}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Finished {split}. Wrote {len(records)} segments"
        f" ({total_duration / 3600:.2f} h) to {output_jsonl}"
    )


if __name__ == "__main__":
    for split in ["test"]:
        process_split(split)
