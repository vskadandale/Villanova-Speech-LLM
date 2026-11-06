"""Score an existing decode per test subset, without re-decoding.

Splits <decode>_gt / <decode>_pred by the corpus and language of each utterance (read from the test jsonl the
decode was made from), writes the per-subset gt/pred files into <out_dir> and scores each one with
src/slam_llm/utils/compute_wer.py, so the numbers use the same normaliser as FINAL_GLOBAL_WER_REPORT.txt.

    python score_by_subset.py <test.jsonl> <decode_prefix> <out_dir>

e.g. <decode_prefix> = exp/ASR_large_en_it_.../asr_epoch_X_step_Y/decode_test. Overwrites files in <out_dir>.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src/slam_llm/utils"))
from compute_wer import compute_wer  # noqa: E402

# Subset of an utterance, recognised from a fragment of its audio path (as in data/large_en_it_enrichment.py)
PATH_SUBSETS = {
    "/Italian_it_": "cv17_it",
    "/English_en_": "cv17_en",
    "/mls_italian/": "mls_it",
    "/LibriSpeech/": "librispeech_en",
    "/fleurs_europe/data/it_it/": "fleurs_it",
    "/fleurs_europe/data/en_us/": "fleurs_en",
    "/mosel_vox_populi/en/": "voxpopuli_en",
    "/italic/": "italic_it",                 # small_en_it
    "/tedlium_segmented/": "tedlium_en",     # small_en_it
}


def subset_of(record):
    names = {name for fragment, name in PATH_SUBSETS.items() if fragment in record["source"]}
    if len(names) != 1:
        raise ValueError(f"Cannot tell the subset of {record['key']}: {record['source']}")
    return names.pop()


def read_lines(path):
    lines = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines[line.split(maxsplit=1)[0]] = line if line.endswith("\n") else line + "\n"
    return lines


def wer_line(report):
    with open(report, encoding="utf-8") as f:
        return next(line.strip() for line in f if line.startswith("%WER"))


def main(test_jsonl, decode_prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(test_jsonl, encoding="utf-8") as f:
        subset = {r["key"]: subset_of(r) for r in map(json.loads, f)}
    gt, pred = read_lines(decode_prefix + "_gt"), read_lines(decode_prefix + "_pred")
    # a partial decode would otherwise give a valid-looking WER on part of the test set (compute_wer only scores the
    # keys present in pred), so the decode has to cover exactly the test set
    if set(gt) != set(subset):
        raise ValueError(f"{decode_prefix}_gt has {len(gt)} keys, {test_jsonl} has {len(subset)} "
                         f"({len(set(gt) - set(subset))} extra, {len(set(subset) - set(gt))} missing): partial decode or wrong jsonl")
    no_pred = set(gt) - set(pred)
    if no_pred:
        raise ValueError(f"{len(no_pred)} keys have no line in {decode_prefix}_pred, e.g. {next(iter(no_pred))}")

    # per corpus, per language (all_en / all_it) and everything (all: should reproduce FINAL_GLOBAL_WER_REPORT.txt)
    groups = defaultdict(list)
    for key in gt:
        name = subset[key]
        groups[name].append(key)
        groups["all_" + name.rsplit("_", 1)[1]].append(key)
        groups["all"].append(key)

    rows = []
    for name in sorted(groups, key=lambda n: (n.startswith("all"), n)):
        keys = groups[name]
        gt_path, pred_path = os.path.join(out_dir, f"{name}_gt"), os.path.join(out_dir, f"{name}_pred")
        with open(gt_path, "w", encoding="utf-8") as g, open(pred_path, "w", encoding="utf-8") as p:
            for key in keys:
                g.write(gt[key])
                if key in pred:
                    p.write(pred[key])
        report = os.path.join(out_dir, f"{name}_WER_REPORT.txt")
        compute_wer(gt_path, pred_path, report)
        rows.append((name, len(keys), wer_line(report)))

    with open(os.path.join(out_dir, "SUMMARY.tsv"), "w", encoding="utf-8") as f:
        f.write("subset\tutterances\twer_line\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    for name, n, line in rows:
        print(f"{name:16s} {n:6d}  {line}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
