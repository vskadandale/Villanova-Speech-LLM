#!/bin/bash
#SBATCH --job-name=fama_meusli_en_it_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          # inference uses 1 dataloader worker; 8 = one GPU's share of a node
#SBATCH --gres=gpu:1               # batch-size-1 decoding on one GPU
#SBATCH --mem=64G                  # LLM + encoder + audio buffers (same as training job)
#SBATCH --time=04:00:00            # decodes voxpopuli_en (1,842 utts, ~22 min before) + small_en_it (4,129 utts, ~50 min)
#SBATCH --account=IscrC_COBRAS
#SBATCH --partition=boost_usr_prod
#SBATCH --output=eval_output_%j.log

# --- Environment Setup ---
module purge
module load cuda/12.2
#module load architecture/expected_arch
#module load nvidia/hpc-sdk

# 3. Force DeepSpeed to recognize the compiler
export DS_BUILD_OPS=0      # Tells DeepSpeed not to try to compile heavy extensions
export DS_SKIP_CUDA_CHECK=1 # Skips the version check that often triggers the /usr/local/cuda error

# Initialize Conda
CONDA_PATH=$WORK/miniforge3
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate vnova

# Emulate Docker's ulimit memlock=-1
ulimit -l unlimited
export PYTHONUNBUFFERED=1

# Execute your specific logic script
bash $WORK/Villanova-Speech-LLM/examples/asr_fama/scripts/eval_meusli_en_it_per_testset.sh
