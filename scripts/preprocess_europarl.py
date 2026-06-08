import os
import json
import subprocess
import pandas as pd

def process_language(lang):
    # Process all three standard splits
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        print(f"--- Processing {lang.upper()} {split.upper()} ---")
        
        base_dir = "/stek/corpora/europarl/v1.1"
        lang_dir = os.path.join(base_dir, lang)
        # Path now includes the split (train/dev/test)
        split_metadata_dir = os.path.join(lang_dir, "en", split)
        source_audio_dir = os.path.join(lang_dir, "audios")
        
        output_audio_dir = f"/stek/corpora/europarl_segmented/{lang}"
        os.makedirs(output_audio_dir, exist_ok=True)

        lst_file = os.path.join(split_metadata_dir, "segments.lst")
        en_text_file = os.path.join(split_metadata_dir, "segments.en")

        if not os.path.exists(lst_file):
            print(f"Split {split} not found for {lang}, skipping...")
            continue

        df = pd.read_csv(lst_file, sep='\s+', header=None, names=["speech_id", "start", "end"])
        
        with open(en_text_file, 'r', encoding='utf-8') as f:
            translations = [" ".join(line.split()) for line in f]

        jsonl_entries = []

        for i, row in df.iterrows():
            duration = row['end'] - row['start']
            
            if duration > 30.0 or duration <= 0:
                continue

            speech_id = row['speech_id']
            
            # Formatting timestamps for the filename (e.g., 8.36 -> 0000836)
            # Multiplying by 100 to keep 2 decimal places as integers
            start_str = f"{int(row['start'] * 100):07d}"
            end_str = f"{int(row['end'] * 100):07d}"

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
                "source": f"/stek/corpora/europarl_segmented/{lang}/{output_file_name}",
                "target": translations[i]
            }
            jsonl_entries.append(entry)

        # Output one JSONL per split
        output_jsonl = f"europarl_{lang}_{split}.jsonl"
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Finished {split}. Created {len(jsonl_entries)} segments.")

# Run for German and French
for language in ['de', 'fr']:
    process_language(language)
