import json
from pathlib import Path

# Builds en/it ASR + ST multitask jsonls from the large_en_it ASR jsonls (with src_lang, from
# large_en_it_enrichment.py) and their translations in large_en_it_st_jsonl (from
# large_en_it_st_preprocessing.py). Both list the same clips in the same order, so every clip appears
# twice, interleaved: as ASR (target = transcript) and as ST (target = translation). Every record gets
# "task" ("asr" or "st"), "src_lang" and "tgt_lang" (= src_lang for ASR), and its key gets the task as
# suffix, since the decode outputs and the WER/BLEU scripts match lines by key.
ASR_DIR = Path("/stek/vskadandale/data/large_en_it_jsonl")
ST_DIR = Path("/stek/vskadandale/data/large_en_it_st_jsonl")
OUTPUT_DIR = Path("/stek/vskadandale/data/large_en_it_multitask_jsonl")

# val = first 1,000 records of validation, in both inputs
SPLITS = ["test", "val", "validation", "train"]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile]


def task_record(record, task, tgt_lang):
    """The record with the task appended to its key, and task, src_lang, tgt_lang as last fields."""
    fields = {k: v for k, v in record.items() if k not in ("src_lang", "tgt_lang")}
    return {**fields, "key": f"{record['key']}_{task}", "task": task, "src_lang": record["src_lang"], "tgt_lang": tgt_lang}


def process_split(split_name):
    output_jsonl = OUTPUT_DIR / f"{split_name}.jsonl"
    if output_jsonl.exists():
        print(f"[{split_name.upper()}] {output_jsonl} already exists, skipping.")
        return

    asr_records = read_jsonl(ASR_DIR / f"{split_name}.jsonl")
    st_records = read_jsonl(ST_DIR / f"{split_name}.jsonl")
    if [r["key"] for r in asr_records] != [r["key"] for r in st_records]:
        raise ValueError(f"[{split_name}] the ASR and ST jsonls do not list the same clips in the same order")
    if any("src_lang" not in r for r in asr_records):
        raise ValueError(f"[{split_name}] ASR records without src_lang: run large_en_it_enrichment.py first")
    if any(asr["src_lang"] != st["src_lang"] for asr, st in zip(asr_records, st_records)):
        raise ValueError(f"[{split_name}] the ASR and ST jsonls disagree on src_lang")

    # Interleaved, so any prefix of the file keeps both tasks in the same proportion
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for asr, st in zip(asr_records, st_records):
            outfile.write(json.dumps(task_record(asr, "asr", asr["src_lang"]), ensure_ascii=False) + "\n")
            outfile.write(json.dumps(task_record(st, "st", st["tgt_lang"]), ensure_ascii=False) + "\n")

    print(f"[{split_name.upper()}] Written {len(asr_records) + len(st_records):,} records "
          f"({len(asr_records):,} ASR + {len(st_records):,} ST) to {output_jsonl}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        process_split(split)
