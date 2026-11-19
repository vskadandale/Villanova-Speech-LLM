#!/bin/bash
set -euo pipefail

# --- Auto-detect Environment ---
if [[ -d "/stek/vskadandale" ]]; then
    export WORK_DIR="/stek/vskadandale"
elif [[ -n "${WORK:-}" ]]; then
    export WORK_DIR="$WORK"
else
    echo "Unknown Environment"
    exit 1
fi

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
USER=${USER:-$(whoami)}

run_dir=$WORK_DIR/Villanova-Speech-LLM
export PYTHONPATH="$run_dir/src"
cd "$run_dir"
recipe_dir=src/slam_llm/recipe
conf_dir=$run_dir/scripts/asr/conf

exp_dir=$run_dir/exp
llm_name=villanova-2b
llm_path="$run_dir/models/$llm_name"
llm_dim=2560
run_name=ASR_cv17_it_fama_scratch_vnova # <task>_<data>_<encoder>_<projector init>_<llm>
prompt_yaml=transcribe_it.yaml
ds_rate=2
learn_rate=1e-4
ctc_layer=24 #16
encoder_projector="pruning"
downsampling_type="linear"
ctc_vocab_size=-1
speech_encoder_pad=0
jsonl_dir=$WORK_DIR/data/cv17_Italian_it_jsonl
encoder_arch=fama
# Everything in this block follows from encoder_arch. The dataset has to match the encoder:
# fama_dataset = fbank+CMVN padded to the longest batch item, speech_dataset = Whisper log-mel padded to 30s.
dataset_type=fama_dataset
encoder_dim=1024
mel_size=80
speech_encoder_path="$run_dir/models/fama-medium-asr"
add_layer_norm=true
# dataset_config.file is what selects the dataset class, dataset_config.dataset is only a label
dataset_file="src/slam_llm/datasets/${dataset_type}.py:get_${dataset_type}"
input_type=mel
pruning_type=none

master_out_dir="$exp_dir/$run_name"
mkdir -p "$master_out_dir"

combined_gt="$master_out_dir/GLOBAL_gt.txt"
combined_pred="$master_out_dir/GLOBAL_pred.txt"

: > "$combined_gt" ; : > "$combined_pred"

# 1. Training (All your specific flags included)
torchrun --nnodes 1 --nproc_per_node 1 "$recipe_dir/finetune.py" \
    --config-path "$conf_dir" --config-name "$prompt_yaml" \
    ++train_config.enable_fsdp=false ++train_config.enable_ddp=true \
    ++train_config.use_fp16=true \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=true \
    ++train_config.peft_config.peft_method=lora \
    ++train_config.batching_strategy=custom \
    ++train_config.warmup_steps=1000 \
    ++train_config.total_steps=1000000 \
    ++train_config.validation_interval=5000 \
    ++train_config.batch_size_training=4 \
    ++train_config.val_batch_size=4 \
    ++train_config.num_workers_dataloader=4 \
    ++train_config.lr=$learn_rate \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path="$llm_path" \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_pad=$speech_encoder_pad \
    ++model_config.encoder_name="$encoder_arch" \
    ++model_config.encoder_path="$speech_encoder_path" \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.add_layer_norm=$add_layer_norm \
    ++model_config.encoder_projector="$encoder_projector" \
    ++model_config.encoder_projector_ds_rate="$ds_rate" \
    ++model_config.pruning_type="$pruning_type" \
    ++model_config.encoder_ctc_layer="$ctc_layer" \
    ++model_config.ctc_vocab_size="$ctc_vocab_size" \
    ++model_config.normalize=true \
    ++model_config.downsampling_type="$downsampling_type" \
    ++dataset_config.dataset=$dataset_type \
    ++dataset_config.file="$dataset_file" \
    ++dataset_config.encoder_projector_ds_rate="$ds_rate" \
    ++dataset_config.train_data_path="$jsonl_dir/train.jsonl" \
    ++dataset_config.val_data_path="$jsonl_dir/val.jsonl" \
    ++dataset_config.input_type=$input_type \
    ++dataset_config.mel_size=$mel_size \
    ++train_config.model_name=asr \
    ++train_config.num_epochs=4 \
    ++metric=acc \
    ++train_config.output_dir="$master_out_dir" \
    ++hydra.run.dir="$master_out_dir" \
    ++log_config.use_wandb=false \
    > "$master_out_dir/train.log" 2>&1

    
# 2. Find Checkpoint (Wildcard to handle step count)
ckpt_path=$(ls -d "$master_out_dir"/asr_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || echo "")

if [[ -d "$ckpt_path" ]]; then
    decode_log="$ckpt_path/decode_test"

    # 3. Inference (Matching your eval script)
    python -u "$recipe_dir/inference_batch.py" \
        --config-path "$conf_dir" --config-name "$prompt_yaml" \
        hydra.run.dir="$ckpt_path" \
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
        ++dataset_config.val_data_path="$jsonl_dir/test.jsonl" \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.use_peft=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++decode_log="$decode_log" \
        ++ckpt_path="$ckpt_path/model.pt" \
        > "$master_out_dir/inference.log" 2>&1
        
        
    # 4. Evaluation
    global_wer_file="$master_out_dir/FINAL_GLOBAL_WER_REPORT.txt"
    python "$run_dir/src/slam_llm/utils/compute_wer.py" "${decode_log}_gt" "${decode_log}_pred" "$global_wer_file"
fi
