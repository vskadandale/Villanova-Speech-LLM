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
code_dir=examples/asr_librispeech

exp_dir=$run_dir/exp
llm_name=eurollm-1.7b
llm_path="$run_dir/models/$llm_name"
llm_dim=2048
prompt_yaml=prompt.yaml
exp_data=voxpopuli
ds_rates=(5)
learn_rate=1e-4
ctc_layer=32
encoder_projector="pruning"
downsampling_type="linear"
ctc_vocab_size=-1
speech_encoder_pad=0
jsonl_dir=$WORK_DIR/voxpopuli_jsonl
encoder_arch=whisper-ctc
encoder_dim=1280
input_type=mel
mel_size=128
speech_encoder_path=$run_dir/models/whisper_ctc_partial_unfreeze.pth
pruning_type=ctc
speech_encoder_prune_thresholds=(1.1 0.9999951124191284 0.999987006187439 0.9999709129333496 0.9999358952045441 0.9998627007007599 0.9931716918945312 0.89990234375)

for ds_rate in "${ds_rates[@]}"; do
    for trained_speech_encoder_prune_thr in "${speech_encoder_prune_thresholds[@]}"; do
        for inference_speech_encoder_prune_thr in "${speech_encoder_prune_thresholds[@]}"; do
            if [[ "$inference_speech_encoder_prune_thr" == "$trained_speech_encoder_prune_thr" ]]; then
                echo "Skipping matched combinations"
                continue
            fi

            # MASTER directory for this unique combination
            exp_tag="${encoder_arch}_${pruning_type}_trained_thr${trained_speech_encoder_prune_thr}_inference_thr${inference_speech_encoder_prune_thr}_ds${ds_rate}"
            master_out_dir="$exp_dir/MISMATCHED_ASR_${exp_data}_${exp_tag}"
            mkdir -p "$master_out_dir"

            combined_gt="$master_out_dir/GLOBAL_gt.txt"
            combined_pred="$master_out_dir/GLOBAL_pred.txt"
            : > "$combined_gt" ; : > "$combined_pred"
            
            # Find Checkpoint Safely
            ckpt_search_pattern="$exp_dir/CTC_ASR_${exp_data}_${encoder_arch}_${pruning_type}_thr${trained_speech_encoder_prune_thr}_ds${ds_rate}/asr_epoch_*_step_*"
            ckpt_path=$(ls -d $ckpt_search_pattern 2>/dev/null | sort -V | tail -n 1 || echo "")
                       
            if [[ -n "$ckpt_path" && -d "$ckpt_path" ]]; then
                # FIX: Keep decode logs inside the unique experiment folder, NOT the shared training folder
                decode_log="$master_out_dir/decode_test"
            
                # 3. Inference
                # FIX: Set hydra.run.dir to master_out_dir to prevent overwriting other inference logs
                python -u "$code_dir/inference_asr_batch.py" \
                    --config-path "conf" --config-name "$prompt_yaml" \
                    hydra.run.dir="$master_out_dir" \
                    ++model_config.llm_name=$llm_name \
                    ++model_config.llm_path="$llm_path" \
                    ++model_config.llm_dim=$llm_dim \
                    ++model_config.encoder_name="$encoder_arch" \
                    ++model_config.encoder_ctc_layer="$ctc_layer" \
                    ++model_config.encoder_projector_ds_rate="$ds_rate" \
                    ++model_config.encoder_prune_thr="$inference_speech_encoder_prune_thr" \
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
            else
                echo "WARNING: No checkpoint folder found matching pattern: $ckpt_search_pattern" | tee -a "$master_out_dir/error.log"
            fi
        done
    done
done
