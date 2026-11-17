#!/bin/bash
set -euo pipefail

# Per-test-set evaluation of the ST checkpoints, on the same test sets as scripts/asr/bilingual/eval_meusli_en_it_per_testset.sh:
#   model=bilingual  finetune_fama_meusli_vnova_bilingual.sh (en->it + it->en ST)
#   model=multitask     finetune_fama_meusli_vnova_multitask.sh    (ASR en/it + ST en->it/it->en)
#
#   1. large_en_it_by_subset: no re-decoding. CV17 Italian (and the other corpora) are part of the large test set,
#      so the existing decode is split by corpus, task and direction and scored.
#   2. voxpopuli_en: decoded. bilingual: en->it ST; multitask: ASR + en->it ST.
#   3. small_en_it (ITALIC it + TED-LIUM en): decoded, multitask only, ASR only (there are no translations of it).
# WER for ASR (compute_wer.py, same normaliser as scripts/eval/score_by_subset.py), BLEU for ST (compute_bleu.py for Italian
# targets, compute_bleu_en.py for English targets, as in the training scripts).
# The ST references are the EuroLLM translations of the test set the model was decoded on: for cv17 it->en those of
# large_en_it_st (96% identical to the separate cv17_Italian_it2en test set), for voxpopuli those of voxpopuli_en2it.
#
# Nothing in the training folder is written to. Everything goes under:
#   exp/EVAL_<training folder name>/<checkpoint name>/
#       large_en_it_by_subset/   per-group gt/pred, <group>_REPORT.txt, SUMMARY.tsv (all_* = the existing FINAL reports)
#       voxpopuli_en/            test.jsonl (tagged with languages), decode, inference logs, .hydra, scores/
#       small_en_it/             same, multitask only
#       RESULTS.tsv              the numbers to report
# Re-running overwrites the contents of this EVAL folder.
# Set ckpt_path=<.../st_epoch_X_step_Y> to pick a checkpoint; default: the latest one, as in the training scripts.

model=${model:-bilingual}

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

# --- Same settings as finetune_fama_meusli_vnova_{bilingual,multitask}.sh ---
case "$model" in
    bilingual) conf_dir=$run_dir/scripts/st/conf; prompt_yaml=prompt_bilingual.yaml; exp_data=large_en_it;           jsonl_dir=$WORK_DIR/data/large_en_it_st_jsonl ;;
    multitask)    conf_dir=$run_dir/scripts/multitask/conf; prompt_yaml=prompt_multitask.yaml;    exp_data=large_en_it_multitask; jsonl_dir=$WORK_DIR/data/large_en_it_multitask_jsonl ;;
    *) echo "model must be bilingual or multitask, not '$model'"; exit 1 ;;
esac
exp_dir=$run_dir/exp
llm_name=villanova-2b
llm_path="$run_dir/models/$llm_name"
llm_dim=2560
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
dataset_file="src/slam_llm/datasets/${dataset_type}.py:get_${dataset_type}"
input_type=mel
meusli_weights_path="$run_dir/models/meusli-villanova-2b.pt"
lora_rank=32
lora_alpha=32
init_skip_prefix=encoder_projector.linear1
pruning_type=none
exp_tag="${encoder_arch}_meusli_${llm_name}"
master_out_dir="$exp_dir/ST_${exp_data}_${exp_tag}"

large_test_jsonl=$jsonl_dir/test.jsonl
vox_asr_jsonl=$WORK_DIR/data/voxpopuli_jsonl/test.jsonl
vox_st_jsonl=$WORK_DIR/data/voxpopuli_en2it_jsonl/test.jsonl
small_test_jsonl=$WORK_DIR/data/small_en_it_jsonl/test.jsonl
helper="scripts/eval/eval_subsets.py"

# --- Checkpoint and inputs (all checked before anything is decoded) ---
ckpt_path=${ckpt_path:-$(ls -d "$master_out_dir"/st_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || echo "")}
if [[ ! -f "$ckpt_path/model.pt" ]]; then
    echo "No checkpoint found (ckpt_path='$ckpt_path')"; exit 1
fi
for f in "$ckpt_path/decode_test_gt" "$ckpt_path/decode_test_pred" "$large_test_jsonl" "$vox_asr_jsonl" "$vox_st_jsonl" "$small_test_jsonl"; do
    [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
done
# every audio file of the test sets to decode must exist
python - "$vox_st_jsonl" "$small_test_jsonl" <<'EOF'
import json, os, sys
for jsonl in sys.argv[1:]:
    with open(jsonl) as f:
        missing = [s for s in (json.loads(l)["source"] for l in f if l.strip()) if not os.path.isfile(s)]
    if missing:
        sys.exit(f"{len(missing)} audio files of {jsonl} not found, e.g. {missing[0]}")
EOF

eval_dir="$exp_dir/EVAL_$(basename "$master_out_dir")/$(basename "$ckpt_path")"
mkdir -p "$eval_dir"
echo "Model:      $model"
echo "Checkpoint: $ckpt_path"
echo "Results:    $eval_dir"

# decode <test jsonl> <output folder>: same flags as the test decode in the training script,
# with the hydra dir, log file and decode files all inside the output folder
decode() {
    local test_jsonl=$1 out_dir=$2
    mkdir -p "$out_dir"
    python -u "$recipe_dir/inference_batch.py" \
        --config-path "$conf_dir" --config-name "$prompt_yaml" \
        hydra.run.dir="$out_dir" \
        ++log_config.log_file="$out_dir/inference_st_batch.log" \
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
        ++train_config.model_name=st \
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

# 1. Split the existing large test decode by corpus, task and direction
large_dir="$eval_dir/large_en_it_by_subset"
python "$helper" score "$large_test_jsonl" "$ckpt_path/decode_test" "$large_dir"

# 2. voxpopuli_en
vox_dir="$eval_dir/voxpopuli_en"
if [[ "$model" == bilingual ]]; then
    python "$helper" tag "$vox_dir/test.jsonl" "$vox_st_jsonl:st:en:it"
else
    python "$helper" tag "$vox_dir/test.jsonl" --key_suffix "$vox_asr_jsonl:asr:en:en" "$vox_st_jsonl:st:en:it"
fi
decode "$vox_dir/test.jsonl" "$vox_dir"
python "$helper" score "$vox_dir/test.jsonl" "$vox_dir/decode_test" "$vox_dir/scores"

# 3. small_en_it, multitask only (ASR)
small_dir="$eval_dir/small_en_it"
if [[ "$model" == multitask ]]; then
    python "$helper" tag "$small_dir/test.jsonl" --key_suffix "$small_test_jsonl:asr:auto:same"
    decode "$small_dir/test.jsonl" "$small_dir"
    python "$helper" score "$small_dir/test.jsonl" "$small_dir/decode_test" "$small_dir/scores"
fi

# 4. Collect the numbers to report: cv17_it and all_* of the large test set, per corpus for the decoded ones
results="$eval_dir/RESULTS.tsv"
{
    printf "checkpoint\t%s\n" "$ckpt_path"
    printf "test_set\tgroup\tutterances\tmetric\tresult\n"
    grep -P "^(cv17_it_|all)" "$large_dir/SUMMARY.tsv" | sed "s/^/$(basename "$jsonl_dir" _jsonl)\t/"
    grep -P "^voxpopuli_en_" "$vox_dir/scores/SUMMARY.tsv" | sed 's/^/voxpopuli_en\t/'
    if [[ "$model" == multitask ]]; then
        grep -P "^(italic_it_|tedlium_en_|all_asr\t)" "$small_dir/scores/SUMMARY.tsv" | sed 's/^/small_en_it\t/'
    fi
} > "$results"
cat "$results"
