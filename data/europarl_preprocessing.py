import os
import json
import subprocess
import pandas as pd

base_dir = "/stek/corpora/europarl/v1.1"
# Layout is {src}/{tgt}/{split}; one pair per source language (there is no en/en)
pair_tgt = {"en": "it", "it": "en"}

def read_split(lang, split):
    # segments.{src} holds the source-language transcripts (for ASR)
    split_metadata_dir = os.path.join(base_dir, lang, pair_tgt[lang], split)
    lst_file = os.path.join(split_metadata_dir, "segments.lst")
    if not os.path.exists(lst_file):
        return None

    df = pd.read_csv(lst_file, sep=r'\s+', header=None, names=["speech_id", "start", "end"])
    with open(os.path.join(split_metadata_dir, f"segments.{lang}"), 'r', encoding='utf-8') as f:
        df["text"] = [" ".join(line.split()) for line in f]

    # Filter before dropping overlaps, so a dropped cut can't block an overlapping good one.
    # Segments under 0.5 s or with punctuation-only text are alignment artifacts.
    duration = df["end"] - df["start"]
    keep = (duration >= 0.5) & (duration <= 30.0) & df["text"].str.contains(r"\w")
    return df[keep].reset_index(drop=True)

def drop_overlaps(df):
    # Some segments overlap others of the same speech (a short cut inside a longer one,
    # or consecutive sentences sharing their edges). Keep shorter cuts first and drop
    # any segment that overlaps one already kept.
    df = df.drop_duplicates(["speech_id", "start", "end"])
    df = df.assign(duration=df["end"] - df["start"]).sort_values("duration")

    kept = []
    for _, group in df.groupby("speech_id", sort=False):
        chosen = []
        for row in group.itertuples():
            if all(row.end <= start or row.start >= end for start, end in chosen):
                chosen.append((row.start, row.end))
                kept.append(row.Index)
    return df.loc[kept].sort_values(["speech_id", "start"]).reset_index(drop=True)

def process_language(lang):
    # Test set only, cut at the original Europarl-ST timestamps
    splits = ['test']

    source_audio_dir = os.path.join(base_dir, lang, "audios")

    for split in splits:
        print(f"--- Processing {lang.upper()} {split.upper()} ---")

        output_audio_dir = f"/stek/corpora/europarl_segmented/{lang}_{split}"
        os.makedirs(output_audio_dir, exist_ok=True)

        df = read_split(lang, split)
        if df is None:
            print(f"Split {split} not found for {lang}, skipping...")
            continue
        df = drop_overlaps(df)

        jsonl_entries = []

        for _, row in df.iterrows():
            duration = row['end'] - row['start']
            speech_id = row['speech_id']
            
            # Formatting timestamps for the filename (e.g., 8.36 -> 0000836)
            # Multiplying by 100 to keep 2 decimal places as integers
            start_str = f"{round(row['start'] * 100):07d}"
            end_str = f"{round(row['end'] * 100):07d}"

            # New Unique ID: SpeechID_Start_End (e.g., en.2008_0000000_0000836)
            unique_id = f"{speech_id}_{start_str}_{end_str}"
            output_file_name = f"{unique_id}.wav"
            output_full_path = os.path.join(output_audio_dir, output_file_name)

            input_file = os.path.join(source_audio_dir, f"{speech_id}.wav")
            if not os.path.exists(input_file):
                input_file = os.path.join(source_audio_dir, f"{speech_id}.m4a")

            if not os.path.exists(output_full_path) and os.path.exists(input_file):
                cmd = [
                    'ffmpeg', '-y', '-ss', str(row['start']), '-t', str(duration),
                    '-i', input_file, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                    output_full_path, '-loglevel', 'error'
                ]
                subprocess.run(cmd)

            entry = {
                "key": unique_id,
                "source": f"/stek/corpora/europarl_segmented/{lang}_{split}/{output_file_name}",
                "target": row['text']
            }
            jsonl_entries.append(entry)

        # Output one JSONL per split
        output_jsonl = f"europarl_{lang}_{split}.jsonl"
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Finished {split}. Created {len(jsonl_entries)} segments.")

# Run for English and Italian
for language in ['en', 'it']:
    process_language(language)
