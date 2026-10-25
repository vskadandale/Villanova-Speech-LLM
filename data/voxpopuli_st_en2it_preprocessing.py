import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Builds en->it speech translation data from the VoxPopuli English ASR jsonls: the same
# records, in the same order, with the English transcript in "target" replaced by its
# Italian translation from EuroLLM-Instruct.
INPUT_DIR = Path("/stek/vskadandale/data/voxpopuli_jsonl")
OUTPUT_DIR = Path("/stek/vskadandale/data/voxpopuli_en2it_jsonl")
# VoxPopuli metadata (asr_{train,dev,test}.tsv) with the original transcripts
TRANSCRIPTS_DIR = Path("/stek/corpora/mosel_vox_populi/en")
# "utter-project/EuroLLM-9B-Instruct" translates better, at about 5x the cost
MODEL_PATH = "/stek/vskadandale/Villanova-Speech-LLM/models/eurollm-1.7b-instruct"

# Splits to process; the small ones first, so they can be inspected while train runs
SPLITS = ["test", "val", "train"]

BATCH_SIZE = 128
MAX_NEW_TOKENS = 256

# Prompt format from the EuroLLM-Instruct model card
PROMPT = "Translate the following English source text to Italian:\nEnglish: {text} \nItalian: "


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Decoder-only models must be left-padded for batched generation
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda", torch.bfloat16)
    model.eval()
    return model, tokenizer


def load_raw_texts():
    """Maps each VoxPopuli utterance id to its original (raw_text) transcript."""
    raw_texts = {}
    for split in ["train", "dev", "test"]:
        with open(TRANSCRIPTS_DIR / f"asr_{split}.tsv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                raw_texts[row["id"]] = row["raw_text"]
    return raw_texts


def english_text(record, raw_texts):
    # The ASR targets are VoxPopuli's normalized_text (lowercased, numbers spelled out).
    # Translate the original raw_text when available, since casing and punctuation
    # help the translation.
    raw_text = " ".join(raw_texts.get(record["key"], "").split())
    return raw_text or record["target"]


@torch.no_grad()
def translate(sentences, model, tokenizer):
    """Translates English sentences to Italian, batching sentences of similar length."""
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    translations = [""] * len(sentences)

    for batch_idx, start in enumerate(range(0, len(order), BATCH_SIZE)):
        batch = order[start:start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(text=sentences[i])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # The model's generation config samples by default; decode greedily instead
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        for i, text in zip(batch, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            # Keep only the translation if the model continues after it
            translations[i] = text.strip().split("\n")[0].strip()

        if batch_idx % 50 == 0:
            print(f"  translated {start + len(batch):,}/{len(order):,}", flush=True)

    return translations


def process_split(split_name, model, tokenizer, raw_texts):
    input_jsonl = INPUT_DIR / f"{split_name}.jsonl"
    output_jsonl = OUTPUT_DIR / f"{split_name}.jsonl"

    if not input_jsonl.exists():
        print(f"{input_jsonl} does not exist.")
        return
    if output_jsonl.exists():
        print(f"[{split_name.upper()}] {output_jsonl} already exists, skipping.")
        return

    with open(input_jsonl, "r", encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile]

    # Translate each distinct sentence once
    texts = [english_text(record, raw_texts) for record in records]
    unique_texts = list(dict.fromkeys(texts))
    print(f"[{split_name.upper()}] Translating {len(unique_texts):,} sentences")
    translation = dict(zip(unique_texts, translate(unique_texts, model, tokenizer)))

    written = empty = copied = 0
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for record, text in zip(records, texts):
            target = translation[text]
            if not target:
                empty += 1
                continue
            copied += target.lower() == text.lower()
            outfile.write(json.dumps({**record, "target": target}, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{split_name.upper()}] Written {written:,} records to {output_jsonl}")
    if empty:
        print(f"  Warning: dropped {empty:,} records with an empty translation")
    if copied:
        print(f"  Warning: {copied:,} translations are identical to the English text")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_texts = load_raw_texts()
    model, tokenizer = load_model()
    for split in SPLITS:
        process_split(split, model, tokenizer, raw_texts)
