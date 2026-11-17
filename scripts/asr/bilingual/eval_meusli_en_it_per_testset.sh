#!/bin/bash
set -euo pipefail

# Per-test-set evaluation of the multilingual (large_en_it) checkpoint trained by finetune_meusli_en_it.sh.
#
#   1. large_en_it_by_subset: no re-decoding. CV17 Italian (and the other large_en_it subsets) are part of the
#      large_en_it test set with the same keys and references, so the existing decode is split per subset and scored.
#   2. voxpopuli_en: not in large_en_it, decoded with the checkpoint (same flags as the large_en_it decode).
#   3. small_en_it (ITALIC it + TED-LIUM en): not in large_en_it, decoded once and scored per subset.
#
# Nothing in the training folder is written to. Everything goes under:
#   exp/EVAL_<training folder name>/<checkpoint name>/
#       large_en_it_by_subset/   per-subset gt/pred, <subset>_WER_REPORT.txt, SUMMARY.tsv (subset "all" = the 10.08 run)
#       voxpopuli_en/            decode_test_gt/_pred, inference logs, .hydra, voxpopuli_en_WER_REPORT.txt
#       small_en_it/             decode_test_gt/_pred, inference logs, .hydra, scores/ (italic_it, tedlium_en, all)
#       RESULTS.tsv              the numbers to report
# Re-running overwrites the contents of this EVAL folder.
# Set ckpt_path=<.../asr_epoch_X_step_Y> to pick a checkpoint; default: the latest one, as in finetune_meusli_en_it.sh.

# --- Auto-detect Environment ---
if [[ -d "/stek/vskadandale" ]]; then
    export WORK_DIR="/stek/vskadandale"
elif [[ -n "${WORK:-}" ]]; then
    export WORK_DIR="$WORK"
else
    echo "Unknown Environment"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

run_dir=$WORK_DIR/Villanova-Speech-LLM
export PYTHONPATH="$run_dir/src"
cd "$run_dir"
recipe_dir=src/slam_llm/recipe
conf_dir=$run_dir/scripts/asr/conf

# --- Same settings as finetune_meusli_en_it.sh ---
exp_dir=$run_dir/exp
llm_name=villanova-2b
llm_path="$run_dir/models/$llm_name"
llm_dim=2560
prompt_yaml=prompt_meusli.yaml
exp_data=large_en_it
ds_rate=2
ctc_layer=24
encoder_projector="linear"
downsampling_type="linear"
ctc_vocab_size=-1
encoder_arch=fama
dataset_type=fama_dataset
encoder_dim=1024
mel_size=80
speech_encoder_path="$run_dir/models/fama-medium-asr"
add_layer_norm=true
dataset_file="src/slam_llm/datasets/${dataset_type}.py:get_${dataset_type}"
input_type=mel
meusli_weights_path="$run_dir/models/meusli-villanova-2b.pt"
lora_rank=32
lora_alpha=32
init_skip_prefix=encoder_projector.linear1
pruning_type=none
exp_tag="${encoder_arch}_meusli_${llm_name}"
master_out_dir="$exp_dir/ASR_${exp_data}_${exp_tag}"

large_test_jsonl=$WORK_DIR/data/large_en_it_jsonl/test.jsonl
vox_test_jsonl=$WORK_DIR/data/voxpopuli_jsonl/test.jsonl
small_test_jsonl=$WORK_DIR/data/small_en_it_jsonl/test.jsonl

# --- Checkpoint and inputs (all checked before anything is decoded) ---
ckpt_path=${ckpt_path:-$(ls -d "$master_out_dir"/asr_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || echo "")}
if [[ ! -f "$ckpt_path/model.pt" ]]; then
    echo "No checkpoint found (ckpt_path='$ckpt_path')"; exit 1
fi
for f in "$ckpt_path/decode_test_gt" "$ckpt_path/decode_test_pred" "$large_test_jsonl" "$vox_test_jsonl" "$small_test_jsonl"; do
    [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
done
# every audio file of the test sets to decode must exist
python - "$vox_test_jsonl" "$small_test_jsonl" <<'EOF'
import json, os, sys
for jsonl in sys.argv[1:]:
    with open(jsonl) as f:
        missing = [s for s in (json.loads(l)["source"] for l in f if l.strip()) if not os.path.isfile(s)]
    if missing:
        sys.exit(f"{len(missing)} audio files of {jsonl} not found, e.g. {missing[0]}")
EOF

eval_dir="$exp_dir/EVAL_$(basename "$master_out_dir")/$(basename "$ckpt_path")"
mkdir -p "$eval_dir"
echo "Checkpoint: $ckpt_path"
echo "Results:    $eval_dir"

# decode <test jsonl> <output folder>: same flags as the large_en_it decode in finetune_meusli_en_it.sh,
# with the hydra dir, log file and decode files all inside the output folder
decode() {
    local test_jsonl=$1 out_dir=$2
    mkdir -p "$out_dir"
    python -u "$recipe_dir/inference_batch.py" \
        --config-path "$conf_dir" --config-name "$prompt_yaml" \
        hydra.run.dir="$out_dir" \
        ++log_config.log_file="$out_dir/inference_asr_batch.log" \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path="$llm_path" \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name="$encoder_arch" \
        ++model_config.encoder_ctc_layer="$ctc_layer" \
        ++model_config.encoder_projector_ds_rate="$ds_rate" \
        ++model_config.ctc_vocab_size="$ctc_vocab_size" \
        ++model_config.encoder_path="$speech_encoder_path" \
        ++model_config.pruning_type="$pruning_type" \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.add_layer_norm=$add_layer_norm \
        ++model_config.encoder_projector="$encoder_projector" \
        ++model_config.normalize=true \
        ++model_config.downsampling_type="$downsampling_type" \
        ++dataset_config.dataset=$dataset_type \
        ++dataset_config.file="$dataset_file" \
        ++dataset_config.encoder_projector_ds_rate="$ds_rate" \
        ++dataset_config.input_type=$input_type \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.val_data_path="$test_jsonl" \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.use_peft=true \
        ++train_config.peft_config.r=$lora_rank \
        ++train_config.peft_config.lora_alpha=$lora_alpha \
        ++model_config.init_ckpt_path="$meusli_weights_path" \
        ++model_config.init_ckpt_skip_prefix=$init_skip_prefix \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++decode_log="$out_dir/decode_test" \
        ++ckpt_path="$ckpt_path/model.pt" \
        > "$out_dir/inference.log" 2>&1
}

# 1. Split the existing large_en_it decode per subset (cv17_it, cv17_en, mls_it, librispeech_en, fleurs_*)
large_dir="$eval_dir/large_en_it_by_subset"
python "scripts/eval/score_by_subset.py" "$large_test_jsonl" "$ckpt_path/decode_test" "$large_dir"

# 2. voxpopuli_en
vox_dir="$eval_dir/voxpopuli_en"
vox_report="$vox_dir/voxpopuli_en_WER_REPORT.txt"
decode "$vox_test_jsonl" "$vox_dir"
python "$run_dir/src/slam_llm/utils/compute_wer.py" "$vox_dir/decode_test_gt" "$vox_dir/decode_test_pred" "$vox_report"

# 3. small_en_it (italic_it + tedlium_en), decoded once, scored per subset
small_dir="$eval_dir/small_en_it"
decode "$small_test_jsonl" "$small_dir"
python "scripts/eval/score_by_subset.py" "$small_test_jsonl" "$small_dir/decode_test" "$small_dir/scores"

# 4. Collect the numbers to report
results="$eval_dir/RESULTS.tsv"
{
    printf "checkpoint\t%s\n" "$ckpt_path"
    printf "test_set\tutterances\twer_line\n"
    grep -P "^(cv17_it|all)\t" "$large_dir/SUMMARY.tsv" | sed 's/^all\t/large_en_it\t/'
    printf "voxpopuli_en\t%s\t%s\n" "$(grep -c . "$vox_dir/decode_test_gt")" "$(grep "^%WER" "$vox_report")"
    grep -P "^(italic_it|tedlium_en|all)\t" "$small_dir/scores/SUMMARY.tsv" | sed 's/^all\t/small_en_it\t/'
} > "$results"
cat "$results"
