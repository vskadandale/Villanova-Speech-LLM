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

run_dir=$WORK_DIR/SLAM-LLM
export PYTHONPATH="$run_dir/src"
cd "$run_dir"
code_dir=examples/ser_iemocap

exp_dir=$run_dir/exp
llm_name=eurollm-1.7b
llm_path="$run_dir/models/$llm_name"
llm_dim=2048
prompt_yaml=prompt.yaml
exp_data=iemocap
ds_rates=(100 50 20 15 10 9 8 7 6 5)
learn_rate=1e-4
ctc_layer=32
encoder_projector="pruning"
downsampling_type="random"
ctc_vocab_size=-1
speech_encoder_pad=0
jsonl_dir=$WORK_DIR/loso_4class
encoder_arch=whisper-ctc 
encoder_dim=1280
input_type=mel
mel_size=128
speech_encoder_path=$run_dir/models/whisper_ctc_partial_unfreeze.pth
pruning_type=none
speech_encoder_prune_thresholds=(1.1)

for ds_rate in "${ds_rates[@]}"; do
    for speech_encoder_prune_thr in "${speech_encoder_prune_thresholds[@]}"; do
        # MASTER directory for this threshold
        exp_tag="${encoder_arch}_${pruning_type}_thr${speech_encoder_prune_thr}_ds${ds_rate}"
        master_out_dir="$exp_dir/RANDOM_SER_${exp_data}_${exp_tag}"
        mkdir -p "$master_out_dir"

        combined_gt="$master_out_dir/GLOBAL_gt.txt"
        combined_pred="$master_out_dir/GLOBAL_pred.txt"
        combined_upv="$master_out_dir/GLOBAL_upv.txt"
        per_fold_summary="$master_out_dir/FOLD_SUMMARY.txt"
        : > "$combined_gt" ; : > "$combined_pred" ; : > "$combined_upv"
        echo "Fold | WA | UWA" > "$per_fold_summary"

        for fold_idx in {1..5}; do
            fold_name="ses${fold_idx}"
            fold_dir="$jsonl_dir/${fold_name}"
            fold_out_dir="$master_out_dir/${fold_name}"
            mkdir -p "$fold_out_dir"

            # 1. Training (All your specific flags included)
            torchrun --nnodes 1 --nproc_per_node 1 "$code_dir/finetune_ser.py" \
                --config-path "conf" --config-name "$prompt_yaml" \
                ++train_config.enable_fsdp=false ++train_config.enable_ddp=true \
                ++train_config.use_fp16=true \
                ++train_config.freeze_encoder=true \
                ++train_config.freeze_llm=true \
                ++train_config.use_peft=true \
                ++train_config.peft_config.peft_method=lora \
                ++train_config.batching_strategy=custom \
                ++train_config.warmup_steps=1000 \
                ++train_config.total_steps=100000 \
                ++train_config.validation_interval=1000 \
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
                ++model_config.encoder_projector="$encoder_projector" \
                ++model_config.encoder_projector_ds_rate="$ds_rate" \
                ++model_config.encoder_prune_thr="$speech_encoder_prune_thr" \
                ++model_config.pruning_type="$pruning_type" \
                ++model_config.encoder_ctc_layer="$ctc_layer" \
                ++model_config.ctc_vocab_size="$ctc_vocab_size" \
                ++model_config.normalize=true \
                ++model_config.downsampling_type="$downsampling_type" \
                ++dataset_config.dataset=speech_dataset \
                ++dataset_config.train_data_path="$fold_dir/train.jsonl" \
                ++dataset_config.val_data_path="$fold_dir/val.jsonl" \
                ++dataset_config.input_type=$input_type \
                ++dataset_config.mel_size=$mel_size \
                ++train_config.model_name=ser \
                ++train_config.num_epochs=10 \
                ++metric=acc \
                ++train_config.output_dir="$fold_out_dir" \
                ++hydra.run.dir="$fold_out_dir" \
                ++log_config.use_wandb=false \
                ++log_config.wandb_entity_name=seraphina-unitn \
                ++log_config.wandb_project_name=slam-ser \
                > "$fold_out_dir/train.log" 2>&1

            # --- THE CLEANUP BRIDGE ---
            echo "Training segment finished. Cleaning up..."
            sleep 5
            pkill -u $USER -9 python || true  # Kill any stuck workers or hung masters
            sleep 2                  # Give the OS a moment to update the file descriptors
            # --------------------------
    
            # 2. Find Checkpoint (Wildcard to handle step count)
            ckpt_path=$(ls -d "$fold_out_dir"/ser_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || echo "")
            
            if [[ -d "$ckpt_path" ]]; then
                decode_log="$ckpt_path/decode_test_fold"
                
                # 3. Inference (Matching your eval script)
                python -u "$code_dir/inference_ser_batch.py" \
                    --config-path "conf" --config-name "$prompt_yaml" \
                    hydra.run.dir="$ckpt_path" \
                    ++model_config.llm_name="$llm_name" \
                    ++model_config.llm_path="$llm_path" \
                    ++model_config.llm_dim=$llm_dim \
                    ++model_config.encoder_name="$encoder_arch" \
                    ++model_config.encoder_ctc_layer="$ctc_layer" \
                    ++model_config.encoder_projector_ds_rate="$ds_rate" \
                    ++model_config.encoder_prune_thr="$speech_encoder_prune_thr" \
                    ++model_config.ctc_vocab_size="$ctc_vocab_size" \
                    ++model_config.encoder_path="$speech_encoder_path" \
                    ++model_config.pruning_type="$pruning_type" \
                    ++model_config.encoder_dim=$encoder_dim \
                    ++model_config.encoder_projector="$encoder_projector" \
                    ++model_config.normalize=true \
                    ++model_config.downsampling_type="$downsampling_type" \
                    ++dataset_config.dataset=speech_dataset \
                    ++dataset_config.input_type=$input_type \
                    ++dataset_config.mel_size=$mel_size \
                    ++dataset_config.val_data_path="$fold_dir/test.jsonl" \
                    ++dataset_config.inference_mode=true \
                    ++train_config.model_name=ser \
                    ++train_config.freeze_encoder=true \
                    ++train_config.freeze_llm=true \
                    ++train_config.use_peft=true \
                    ++train_config.batching_strategy=custom \
                    ++train_config.num_epochs=1 \
                    ++train_config.val_batch_size=1 \
                    ++decode_log="$decode_log" \
                    ++ckpt_path="$ckpt_path/model.pt" \
                    > "$fold_out_dir/inference.log" 2>&1
                    
                # 4. Evaluation
                gt_file="${decode_log}_gt"
                pred_file="${decode_log}_pred"
                upv_file="${decode_log}_upv"
                fold_cm_csv="${fold_out_dir}/confusion_matrix.csv"
                if [[ -f "$gt_file" ]]; then
                    eval_out=$(python $run_dir/src/slam_llm/utils/compute_ser_accuracy.py \
                    --gt_file $gt_file \
                    --pred_file "$pred_file" \
                    --cm_path "$fold_cm_csv")
                    # Extract metrics for summary table
                    wa=$(echo "$eval_out" | grep "Weighted Accuracy (WA):" | awk '{print $4}')
                    uwa=$(echo "$eval_out" | grep "Unweighted Accuracy (UWA):" | awk '{print $4}')
                    
                    # Save the fold-wise results to a summary file
                    echo "$fold_name | $wa | $uwa" >> "$per_fold_summary"
                    
                    # Store the full text output in the fold dir just in case
                    echo "$eval_out" > "$fold_out_dir/eval_report.txt"

                    # Append for Global (Pooled) metrics
                    cat "$gt_file" >> "$combined_gt"
                    cat "$pred_file" >> "$combined_pred"
                    cat "$upv_file" >> "$combined_upv"
                fi
            fi
        done
        # 5. Global Eval (Pooled)
        python "$run_dir/src/slam_llm/utils/compute_ser_accuracy.py" --gt_file "$combined_gt" --pred_file "$combined_pred" \
            > "$master_out_dir/FINAL_GLOBAL_REPORT.txt"
    done
done