#!/bin/bash
#SBATCH --job-name=ctc_iemocap_prune_threshold_search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16         # Double the dataloader workers for overhead
#SBATCH --gres=gpu:1               # Only 1 GPU needed for this search
#SBATCH --mem=64G                  # Plenty of room for LLM + audio buffers
#SBATCH --time=76:00:00            # Binary search is slow; give it 24h
#SBATCH --account=IscrC_MLMA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod      # Necessary for a 24h limit
#SBATCH --output=search_output_%j.log

# --- Environment Setup ---
module purge
module load cuda/12.1
#module load architecture/expected_arch
#module load nvidia/hpc-sdk

# 3. Force DeepSpeed to recognize the compiler
export DS_BUILD_OPS=0      # Tells DeepSpeed not to try to compile heavy extensions
export DS_SKIP_CUDA_CHECK=1 # Skips the version check that often triggers the /usr/local/cuda error

# Initialize Conda
CONDA_PATH=$WORK/miniconda3
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate tok-pruning

# Emulate Docker's ulimit memlock=-1
ulimit -l unlimited
export PYTHONUNBUFFERED=1

# Execute your specific logic script
bash $WORK/SLAM-LLM/examples/asr_librispeech/scripts/prune_threshold_hunter_ctc.sh
