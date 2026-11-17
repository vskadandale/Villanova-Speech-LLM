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
conf_dir=$run_dir/scripts/st/conf

exp_dir=$run_dir/exp
llm_name=villanova-2b
llm_path="$run_dir/models/$llm_name"
llm_dim=2560
prompt_yaml=prompt_bilingual.yaml # one prompt per direction, picked by each record's tgt_lang
exp_data=large_en_it
ds_rate=2
learn_rate=1e-4
ctc_layer=24 #16
encoder_projector="linear"
downsampling_type="linear"
ctc_vocab_size=-1
speech_encoder_pad=0

jsonl_dir=$WORK_DIR/data/large_en_it_st_jsonl
encoder_arch=fama # fama | whisper-ctc
# Everything in this block follows from encoder_arch. The dataset has to match the encoder:
# fama_dataset = fbank+CMVN padded to the longest batch item, speech_dataset = Whisper log-mel padded to 30s.
dataset_type=fama_dataset
encoder_dim=1024
mel_size=80
speech_encoder_path="$run_dir/models/fama-medium-asr"

# dataset_config.file is what selects the dataset class, dataset_config.dataset is only a label
dataset_file="src/slam_llm/datasets/${dataset_type}.py:get_${dataset_type}"
input_type=mel

meusli_weights_path="$run_dir/models/meusli-villanova-2b.pt" # = HF SpeechTek/MEUSLI_projector_v2: projector 6400->2048->2560 + LoRA on villanova-2b
lora_rank=32    # rank of the LoRA stored in that file (q_proj, v_proj on all 18 layers)
lora_alpha=32  # not stored in the file; repo default, which MEUSLI was trained with
init_skip_prefix=encoder_projector.linear1  # 6400 = 5 Whisper frames x 1280: not loaded, re-initialised as 1024*ds_rate -> 2048
train_only_prefix=null #encoder_projector.linear1 # train only the new layer, MEUSLI's linear2 and LoRA stay frozen; null = train them too
pruning_type=none

exp_tag="${encoder_arch}_meusli_${llm_name}"
master_out_dir="$exp_dir/ST_${exp_data}_${exp_tag}"
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
    ++train_config.peft_config.r=$lora_rank \
    ++train_config.peft_config.lora_alpha=$lora_alpha \
    ++train_config.train_only_prefix=$train_only_prefix \
    ++model_config.init_ckpt_path="$meusli_weights_path" \
    ++model_config.init_ckpt_skip_prefix=$init_skip_prefix \
    ++train_config.batching_strategy=custom \
    ++train_config.warmup_steps=1000 \
    ++train_config.total_steps=1000000 \
    ++train_config.validation_interval=5000 \
    ++train_config.batch_size_training=2 \
    ++train_config.gradient_accumulation_steps=2 \
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
    ++train_config.model_name=st \
    ++train_config.num_epochs=4 \
    ++metric=acc \
    ++train_config.output_dir="$master_out_dir" \
    ++hydra.run.dir="$master_out_dir" \
    ++log_config.use_wandb=false \
    ++log_config.wandb_entity_name=seraphina-unitn \
    ++log_config.wandb_project_name=slam-st \
    > "$master_out_dir/train.log" 2>&1

#++ckpt_path="$meusli_weights_path" \
    
# 2. Find Checkpoint (Wildcard to handle step count)
ckpt_path=$(ls -d "$master_out_dir"/st_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1 || echo "")
#ckpt_path=$master_out_dir

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
        ++decode_log="$decode_log" \
        ++ckpt_path="$ckpt_path/model.pt" \
        > "$master_out_dir/inference.log" 2>&1
        
    #++ckpt_path="$meusli_weights_path" \
        
    # 4. Evaluation (BLEU instead of WER), per direction: the test set mixes en->it and it->en
    # split decode_test_{gt,pred} into decode_test_{it,en}_{gt,pred} by each record's tgt_lang
    python - "$jsonl_dir/test.jsonl" "$decode_log" <<'EOF'
import json, sys
jsonl, decode_log = sys.argv[1:]
with open(jsonl, encoding="utf-8") as f:
    tgt_langs = {r["key"]: r["tgt_lang"] for r in map(json.loads, f)}
for suffix in ["_gt", "_pred"]:
    with open(decode_log + suffix, encoding="utf-8") as fin:
        lines = fin.readlines()
    for lang in sorted(set(tgt_langs.values())):
        with open(f"{decode_log}_{lang}{suffix}", "w", encoding="utf-8") as fout:
            fout.writelines(line for line in lines if tgt_langs.get(line.split("\t", 1)[0].strip()) == lang)
EOF
    python "$run_dir/src/slam_llm/utils/compute_bleu.py" "${decode_log}_it_gt" "${decode_log}_it_pred" "$master_out_dir/FINAL_GLOBAL_BLEU_REPORT_en2it.txt"
    python "$run_dir/src/slam_llm/utils/compute_bleu_en.py" "${decode_log}_en_gt" "${decode_log}_en_pred" "$master_out_dir/FINAL_GLOBAL_BLEU_REPORT_it2en.txt"
fi