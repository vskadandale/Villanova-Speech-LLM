#!/bin/bash
export PATH=$PATH:~/bin

EXP_BASE_DIR="$WORK/SLAM-LLM/exp"
REMOTE="Personal OneDrive"
BASE_DEST="Weights/tok-pruning/With_Finetuned_Whisper_CTC/SER/iemocap-5fcv"

# Use find with -print0 and read -d '' to safely handle all directories
find "$EXP_BASE_DIR" -maxdepth 1 -type d -name "*thr*" -print0 | sort -zV | while read -d '' -r full_path; do

    dir_name=$(basename "$full_path")

    # 1. Determine Subfolder
    if [[ "$dir_name" == COSD* ]]; then SUB="cosd"
    elif [[ "$dir_name" == CTC* ]]; then SUB="ctc"
    elif [[ "$dir_name" == SER* ]]; then SUB="var_ds"
    else SUB="base"; fi

    SUFFIX=$(echo "$dir_name" | grep -o "thr.*")
    DEST_PATH="$REMOTE:$BASE_DEST/$SUB/$SUFFIX"

    echo "=========================================================="
    echo "ROOT: $dir_name"

    # 2. Upload Global Files
    for g_file in "FINAL_GLOBAL_REPORT.txt" "FOLD_SUMMARY.txt" "GLOBAL_gt.txt" "GLOBAL_pred.txt" "GLOBAL_upv.txt"; do
        if [ -f "$full_path/$g_file" ]; then
            rclone copy -P "$full_path/$g_file" "$DEST_PATH/"
        fi
    done

    # 3. Process Sessions - Using a robust pipe to ensure all ses1-5 are caught
    found_ses=false
    while read -r ses_path; do
        [ -z "$ses_path" ] && continue
        found_ses=true

        ses_name=$(basename "$ses_path")
        CURRENT_DEST="$DEST_PATH/$ses_name"

        echo "  --> Session: $ses_name"

        # Upload session metrics
        for s_file in "confusion_matrix.csv" "eval_report.txt"; do
            if [ -f "$ses_path/$s_file" ]; then
                rclone copy -P "$ses_path/$s_file" "$CURRENT_DEST/"
            fi
        done

        # Find latest checkpoint
        ckpt=$(ls -d "$ses_path"/{asr,ser}_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$ckpt" ]; then
            echo "      Uploading: $(basename "$ckpt")"
            rclone copy -P "$ckpt" "$CURRENT_DEST/$(basename "$ckpt")"
        fi

    done < <(ls -d "$full_path"/ses*/ 2>/dev/null | sort -V)

    # 4. Fallback if no sessions
    if [ "$found_ses" = false ]; then
        ckpt=$(ls -d "$full_path"/{asr,ser}_epoch_*_step_* 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$ckpt" ]; then
            rclone copy -P "$ckpt" "$DEST_PATH/$(basename "$ckpt")"
        fi
    fi
done

