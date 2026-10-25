import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Builds en<->it speech translation data from the large_en_it ASR jsonls (CV17, FLEURS, LibriSpeech,
# MLS): the same records, in the same order, with the transcript in "target" replaced by its
# translation into the other language from EuroLLM-Instruct. English audio gets an Italian target and
# Italian audio an English one; "src_lang" and "tgt_lang" record the direction of each record.
INPUT_DIR = Path("/stek/vskadandale/data/large_en_it_jsonl")
OUTPUT_DIR = Path("/stek/vskadandale/data/large_en_it_st_jsonl")
# FLEURS metadata ({train,dev,test}.tsv per language) with the original transcripts
FLEURS_DIR = Path("/stek/corpora/fleurs_europe/data")
# "utter-project/EuroLLM-9B-Instruct" translates better, at about 5x the cost
MODEL_PATH = "/stek/vskadandale/Villanova-Speech-LLM/models/eurollm-1.7b-instruct"

# Splits to process; the small ones first, so they can be inspected while train runs
SPLITS = ["test", "val", "train"]

BATCH_SIZE = 128
MAX_NEW_TOKENS = 256

# Language of the audio, recognised from a fragment of its path
PATH_LANGS = {
    "/English_en_": "en",                # CV17 English
    "/fleurs_europe/data/en_us/": "en",
    "/LibriSpeech/": "en",
    "/Italian_it_": "it",                # CV17 Italian
    "/fleurs_europe/data/it_it/": "it",
    "/mls_italian/": "it",
}
TARGET_LANGS = {"en": "it", "it": "en"}
LANG_NAMES = {"en": "English", "it": "Italian"}

# Prompt format from the EuroLLM-Instruct model card
PROMPT = "Translate the following {src} source text to {tgt}:\n{src}: {text} \n{tgt}: "


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Decoder-only models must be left-padded for batched generation
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda", torch.bfloat16)
    model.eval()
    return model, tokenizer


def load_fleurs_raw_texts():
    """Maps each FLEURS (language dir, audio file name) to its original (raw_transcription) transcript."""
    raw_texts = {}
    for lang_dir in ["en_us", "it_it"]:
        for split in ["train", "dev", "test"]:
            with open(FLEURS_DIR / lang_dir / f"{split}.tsv", "r", encoding="utf-8") as f:
                # no header; columns: id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
                for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    raw_texts[(lang_dir, row[1])] = row[2]
    return raw_texts


def source_language(record):
    langs = {lang for fragment, lang in PATH_LANGS.items() if fragment in record["source"]}
    if len(langs) != 1:
        raise ValueError(f"Cannot tell the language of {record['source']}")
    return langs.pop()


def source_text(record, fleurs_raw_texts):
    # The ASR targets are lowercased. Translate the original transcript where the corpus has one,
    # since casing and punctuation help the translation: CV17 stores it next to the audio, FLEURS in
    # its metadata. LibriSpeech and MLS only have the normalized text, which is translated as is.
    audio_path = Path(record["source"])
    if "/common_voice_17_europe/" in record["source"]:
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            return " ".join(txt_path.read_text(encoding="utf-8").split())
    elif "/fleurs_europe/" in record["source"]:
        # .../data/{lang_dir}/audio/{split}/{file_name}
        raw_text = " ".join(fleurs_raw_texts.get((audio_path.parents[2].name, audio_path.name), "").split())
        if raw_text:
            return raw_text
    return record["target"]


@torch.no_grad()
def translate(sentences, src_lang, model, tokenizer):
    """Translates sentences from src_lang to the other language, batching sentences of similar length."""
    src, tgt = LANG_NAMES[src_lang], LANG_NAMES[TARGET_LANGS[src_lang]]
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    translations = [""] * len(sentences)

    for batch_idx, start in enumerate(range(0, len(order), BATCH_SIZE)):
        batch = order[start:start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(src=src, tgt=tgt, text=sentences[i])}],
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


def process_split(split_name, model, tokenizer, fleurs_raw_texts):
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

    langs = [source_language(record) for record in records]
    texts = [source_text(record, fleurs_raw_texts) for record in records]

    # Translate each distinct sentence once per direction
    translation = {}
    for src_lang, tgt_lang in TARGET_LANGS.items():
        unique_texts = list(dict.fromkeys(text for text, lang in zip(texts, langs) if lang == src_lang))
        print(f"[{split_name.upper()}] Translating {len(unique_texts):,} {LANG_NAMES[src_lang]} sentences to {LANG_NAMES[tgt_lang]}")
        translation[src_lang] = dict(zip(unique_texts, translate(unique_texts, src_lang, model, tokenizer)))

    written = empty = copied = 0
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        for record, lang, text in zip(records, langs, texts):
            target = translation[lang][text]
            if not target:
                empty += 1
                continue
            copied += target.lower() == text.lower()
            outfile.write(json.dumps({**record, "target": target, "src_lang": lang, "tgt_lang": TARGET_LANGS[lang]}, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{split_name.upper()}] Written {written:,} records to {output_jsonl}")
    if empty:
        print(f"  Warning: dropped {empty:,} records with an empty translation")
    if copied:
        print(f"  Warning: {copied:,} translations are identical to the source text")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fleurs_raw_texts = load_fleurs_raw_texts()
    model, tokenizer = load_model()
    for split in SPLITS:
        process_split(split, model, tokenizer, fleurs_raw_texts)
