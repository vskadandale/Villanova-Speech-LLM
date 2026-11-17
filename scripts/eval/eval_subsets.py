"""Helpers for eval_fama_meusli_vnova_per_testset.sh (multilingual ST and multitask ASR+ST checkpoints).

tag: write a jsonl that the multilingual / multitask prompts can decode, from test sets without language fields
    python eval_subsets.py tag <out.jsonl> [--key_suffix] <jsonl>:<task>:<src_lang>:<tgt_lang> ...
    every record of <jsonl> gets task, src_lang, tgt_lang (src_lang "auto" = from the corpus of its audio path,
    tgt_lang "same" = src_lang); --key_suffix appends _<task> to the keys, as in large_en_it_multitask_jsonl.

score: split a decode by corpus, task and direction and score each group: WER for ASR (compute_wer.py,
BasicTextNormalizer, as in score_by_subset.py), BLEU for ST (compute_bleu.py for Italian targets,
compute_bleu_en.py for English targets, as in the training scripts)
    python eval_subsets.py score <test.jsonl> <decode_prefix> <out_dir>
    groups: <corpus>_<task>_<src>2<tgt>, all_<task>_<src>2<tgt> per direction, and all_asr (both ASR languages)
    out_dir gets <group>_{gt,pred}, <group>_REPORT.txt and SUMMARY.tsv; existing files are overwritten.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/slam_llm/utils")

# corpus and language of an utterance, from a fragment of its audio path
PATH_CORPORA = {
    "/Italian_it_": ("cv17_it", "it"),
    "/English_en_": ("cv17_en", "en"),
    "/mls_italian/": ("mls_it", "it"),
    "/LibriSpeech/": ("librispeech_en", "en"),
    "/fleurs_europe/data/it_it/": ("fleurs_it", "it"),
    "/fleurs_europe/data/en_us/": ("fleurs_en", "en"),
    "/mosel_vox_populi/en/": ("voxpopuli_en", "en"),
    "/italic/": ("italic_it", "it"),
    "/tedlium_segmented/": ("tedlium_en", "en"),
}


def corpus_of(record):
    found = {v for fragment, v in PATH_CORPORA.items() if fragment in record["source"]}
    if len(found) != 1:
        raise ValueError(f"Cannot tell the corpus of {record['key']}: {record['source']}")
    return found.pop()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tag(out_jsonl, *specs):
    key_suffix = "--key_suffix" in specs
    records = []
    for spec in (s for s in specs if s != "--key_suffix"):
        jsonl, task, src, tgt = spec.rsplit(":", 3)
        for r in read_jsonl(jsonl):
            s = corpus_of(r)[1] if src == "auto" else src
            t = s if tgt == "same" else tgt
            key = f"{r['key']}_{task}" if key_suffix else r["key"]
            records.append({**r, "key": key, "task": task, "src_lang": s, "tgt_lang": t})
    keys = [r["key"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{out_jsonl}: duplicate keys (use --key_suffix when mixing tasks)")
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out_jsonl}: {len(records)} records")


def read_decode(path):
    lines = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines[line.split("\t", 1)[0].strip()] = line if line.endswith("\n") else line + "\n"
    return lines


def score(test_jsonl, decode_prefix, out_dir):
    sys.path.insert(0, UTILS)
    from compute_wer import compute_wer

    os.makedirs(out_dir, exist_ok=True)
    info = {}
    for r in read_jsonl(test_jsonl):
        task = r.get("task") or ("asr" if r["src_lang"] == r["tgt_lang"] else "st")
        info[r["key"]] = (corpus_of(r)[0], task, f"{r['src_lang']}2{r['tgt_lang']}", r["tgt_lang"])
    gt, pred = read_decode(decode_prefix + "_gt"), read_decode(decode_prefix + "_pred")
    if set(gt) != set(info):
        raise ValueError(f"{decode_prefix}_gt has {len(gt)} keys, {test_jsonl} has {len(info)}: partial decode or wrong jsonl")
    no_pred = set(gt) - set(pred)
    if no_pred:
        raise ValueError(f"{len(no_pred)} keys have no line in {decode_prefix}_pred, e.g. {next(iter(no_pred))}")

    groups = defaultdict(list)
    for key in gt:
        corpus, task, direction, tgt_lang = info[key]
        groups[(f"{corpus}_{task}_{direction}", task, tgt_lang)].append(key)
        groups[(f"all_{task}_{direction}", task, tgt_lang)].append(key)
        if task == "asr":
            groups[("all_asr", "asr", None)].append(key)

    rows = []
    for (name, task, tgt_lang), keys in sorted(groups.items(), key=lambda g: (g[0][0].startswith("all"), g[0][0])):
        gt_path, pred_path = os.path.join(out_dir, f"{name}_gt"), os.path.join(out_dir, f"{name}_pred")
        with open(gt_path, "w", encoding="utf-8") as g, open(pred_path, "w", encoding="utf-8") as p:
            for key in keys:
                g.write(gt[key])
                p.write(pred[key])
        report = os.path.join(out_dir, f"{name}_REPORT.txt")
        if task == "asr":
            compute_wer(gt_path, pred_path, report)
            with open(report, encoding="utf-8") as f:
                result = next(line.strip() for line in f if line.startswith("%WER"))
            metric = "WER"
        else:
            scorer = "compute_bleu_en.py" if tgt_lang == "en" else "compute_bleu.py"
            subprocess.run([sys.executable, os.path.join(UTILS, scorer), gt_path, pred_path, report], check=True)
            with open(report, encoding="utf-8") as f:
                result = " | ".join(line.strip() for line in f if "BLEU" in line)
            if not result:
                raise RuntimeError(f"{scorer} wrote no BLEU into {report}")
            metric = "BLEU"
        rows.append((name, len(keys), metric, result))

    with open(os.path.join(out_dir, "SUMMARY.tsv"), "w", encoding="utf-8") as f:
        f.write("group\tutterances\tmetric\tresult\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    for name, n, metric, result in rows:
        print(f"{name:28s} {n:6d}  {result}")


if __name__ == "__main__":
    commands = {"tag": tag, "score": score}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:])
