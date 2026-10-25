import json
import os
import shutil
from pathlib import Path

# Adds "src_lang" (the language of the audio, "en" or "it") to every record of the large_en_it ASR
# jsonls, in place. The language is recognised from the audio path, as in large_en_it_st_preprocessing.py.
DATA_DIR = Path("/stek/vskadandale/data/large_en_it_jsonl")

# Language of the audio, recognised from a fragment of its path
PATH_LANGS = {
    "/English_en_": "en",                # CV17 English
    "/fleurs_europe/data/en_us/": "en",
    "/LibriSpeech/": "en",
    "/Italian_it_": "it",                # CV17 Italian
    "/fleurs_europe/data/it_it/": "it",
    "/mls_italian/": "it",
}


def source_language(record):
    langs = {lang for fragment, lang in PATH_LANGS.items() if fragment in record["source"]}
    if len(langs) != 1:
        raise ValueError(f"Cannot tell the language of {record['source']}")
    return langs.pop()


def enrich(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile]

    langs = [source_language(record) for record in records]
    conflicts = [record["key"] for record, lang in zip(records, langs) if record.get("src_lang", lang) != lang]
    if conflicts:
        raise ValueError(f"{jsonl_path}: {len(conflicts):,} records already have a different src_lang, e.g. {conflicts[0]}")
    if all("src_lang" in record for record in records):
        print(f"[{jsonl_path.name}] already tagged, skipping.")
        return

    # Write next to the original and swap, so an interrupted run leaves the file intact
    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as outfile:
        for record, lang in zip(records, langs):
            outfile.write(json.dumps({**record, "src_lang": lang}, ensure_ascii=False) + "\n")
    shutil.copymode(jsonl_path, tmp_path)
    os.replace(tmp_path, jsonl_path)

    counts = {lang: langs.count(lang) for lang in sorted(set(langs))}
    print(f"[{jsonl_path.name}] Tagged {len(records):,} records: {counts}")


if __name__ == "__main__":
    for jsonl_path in sorted(DATA_DIR.glob("*.jsonl")):
        enrich(jsonl_path)
