#!/bin/bash

export PATH=$PATH:~/vskadandale/bin
# Configuration

# Paths
EXP_BASE_DIR="/raid/home/stek/vskadandale/SLAM-LLM/exp"
REMOTE="Personal OneDrive"
BASE_DEST="Weights/tok-pruning/With_Finetuned_Whisper_CTC/ASR/voxpopuli-test"

# Check if the directory exists before starting
if [ ! -d "$EXP_BASE_DIR" ]; then
    echo "Error: Directory $EXP_BASE_DIR does not exist."
    exit 1
fi

for full_path in "$EXP_BASE_DIR"/ASR*/; do
    # Remove trailing slash
    full_path=${full_path%/}

    # Get just the folder name (e.g., ASR_WHISPER_CTC_...)
    dir_name=$(basename "$full_path")

    # 1. Determine Subfolder (cosd, ctc, or base)
    if [[ "$dir_name" == COSD* ]]; then
        SUB="cosd"
    elif [[ "$dir_name" == CTC* ]]; then
        SUB="ctc"
    elif [[ "$dir_name" == ASR* ]]; then
        SUB="var_ds"	     
    else
        SUB="base"
    fi

    # 2. Extract Threshold Suffix (e.g., thr1.1_ds5)
    SUFFIX=$(echo "$dir_name" | grep -o "thr.*")

    if [ -z "$SUFFIX" ]; then
        continue # Skip folders that don't match the thr pattern
    fi

    DEST_PATH="$REMOTE:$BASE_DEST/$SUB/$SUFFIX"

    echo "--- Processing: $dir_name ---"

    # 3. Find the latest checkpoint using your logic
    # We use the full path to locate the directories
    ckpt_full_path=$(ls -d "$full_path"/asr_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1)

    if [ -n "$ckpt_full_path" ]; then
        ckpt_name=$(basename "$ckpt_full_path")

        # Copy Report
        if [ -f "$full_path/FINAL_GLOBAL_WER_REPORT.txt" ]; then
            rclone copy -P "$full_path/FINAL_GLOBAL_WER_REPORT.txt" "$DEST_PATH/"
        fi

        # Copy Checkpoint Folder
        echo "Uploading $ckpt_name to $DEST_PATH"
        rclone copy -P "$ckpt_full_path" "$DEST_PATH/$ckpt_name"
    else
        echo "No checkpoint found in $full_path"
    fi
done
