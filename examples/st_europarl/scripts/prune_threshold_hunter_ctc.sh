#!/bin/bash
set -uo pipefail

# --- Configuration Targets ---
TARGET_PERCENTAGES=(16.67 28.57 37.5 44.44 50.0 66.67 75.0 90.0 95.0)
TOLERANCE=0.001  # Tight tolerance for pruning accuracy
MAX_ITERS=25
INITIAL_LOW=0  # CTC thresholds usually range from 0.0 to 1.0
INITIAL_HIGH=1

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
code_dir=examples/st_europarl

# --- Model & Path Config ---
llm_name=eurollm-1.7b
llm_path="$run_dir/models/$llm_name"
llm_dim=2048
exp_dir=$run_dir/exp
prompt_yaml=prompt.yaml
exp_data=europarl
ds_rate=5
learn_rate=1e-4
ctc_layer=32
encoder_projector="pruning"
ctc_vocab_size=-1
speech_encoder_pad=0
jsonl_dir=$WORK_DIR/europarl_de2en_fr2en_jsonl
encoder_arch=whisper-ctc 
encoder_dim=1280
input_type=mel
mel_size=128
speech_encoder_path=$run_dir/models/whisper_ctc_partial_unfreeze.pth
pruning_type=ctc
ckpt_path="$run_dir/models"

# Create a small probe for speed (500 samples is plenty for thresholding)
PROBE_JSONL="$jsonl_dir/test.jsonl"

FINAL_RESULTS=""

for TARGET_PCT in "${TARGET_PERCENTAGES[@]}"; do
    LOW=$INITIAL_LOW
    HIGH=$INITIAL_HIGH

    for (( i=1; i<=MAX_ITERS; i++ )); do
        CURRENT_THR=$(python3 -c "print(($LOW + $HIGH) / 2)")
        
        echo -n "Target $TARGET_PCT% | Iter $i: Thr $CURRENT_THR -> "

        torchrun --nnodes 1 --nproc_per_node 1 "$code_dir/finetune_st.py" \
            --config-path "conf" --config-name "$prompt_yaml" \
            ++train_config.enable_fsdp=false ++train_config.enable_ddp=true \
            ++train_config.use_fp16=true \
            ++train_config.freeze_encoder=true \
            ++train_config.freeze_llm=true \
            ++train_config.use_peft=true \
            ++train_config.peft_config.peft_method=lora \
            ++train_config.batching_strategy=custom \
            ++train_config.batch_size_training=1 \
            ++train_config.num_workers_dataloader=8 \
            ++train_config.val_batch_size=1 \
            ++train_config.run_validation=false \
            ++train_config.lr=$learn_rate \
            ++model_config.llm_name=$llm_name \
            ++model_config.llm_path="$llm_path" \
            ++model_config.llm_dim=$llm_dim \
            ++model_config.encoder_pad=$speech_encoder_pad \
            ++model_config.encoder_name="$encoder_arch" \
            ++model_config.encoder_path="$speech_encoder_path" \
            ++model_config.encoder_dim=$encoder_dim \
            ++model_config.encoder_projector="$encoder_projector" \
            ++model_config.encoder_projector_ds_rate=$ds_rate \
            ++model_config.encoder_prune_thr=$CURRENT_THR \
            ++model_config.pruning_type="$pruning_type" \
            ++model_config.encoder_ctc_layer="$ctc_layer" \
            ++model_config.ctc_vocab_size="$ctc_vocab_size" \
            ++model_config.normalize=true \
            ++dataset_config.dataset=speech_dataset \
            ++dataset_config.train_data_path=$PROBE_JSONL \
            ++dataset_config.val_data_path=$PROBE_JSONL \
            ++dataset_config.input_type=$input_type \
            ++dataset_config.mel_size=$mel_size \
            ++train_config.model_name=st \
            ++train_config.num_epochs=1 \
            ++metric=acc \
            > "$run_dir/search_temp_ctc.log" 2>&1

        # --- NEW: GRACEFUL BRIDGE ---
        # Capture the exit code of torchrun
        EXIT_CODE=$?
        
        # Clean up workers immediately so they don't sit in memory
        pkill -u $USER -9 python || true
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Error: torchrun failed at Iter $i. Check search_temp_ctc.log"
            exit 1
        fi

        # Extract Pruning Percentage
        RESULT_PCT=$(python "$run_dir/src/slam_llm/utils/compute_prune_percentage.py" "$run_dir/search_temp_ctc.log" | grep "Micro Average" | awk '{print $3}' | tr -d '%')
        
        if [[ -z "$RESULT_PCT" ]]; then
            echo "Error: Check search_temp_ctc.log"; exit 1
        fi
        echo "$RESULT_PCT%"
        
        # Binary Search Logic
        DIFF=$(python3 -c "print(abs($RESULT_PCT - $TARGET_PCT))")
        if [ "$(python3 -c "print(1 if $DIFF < $TOLERANCE else 0)")" -eq 1 ]; then
            break
        fi

        if [ "$(python3 -c "print(1 if $RESULT_PCT < $TARGET_PCT else 0)")" -eq 1 ]; then
            HIGH=$CURRENT_THR
        else
            LOW=$CURRENT_THR
        fi
    done
    FINAL_RESULTS+="\nTarget: $TARGET_PCT% | Found Threshold: $CURRENT_THR"
done

echo -e "\n\n-------------------------------------------------------"
echo "SEARCH COMPLETE"
echo -e "$FINAL_RESULTS"
echo "-------------------------------------------------------"

