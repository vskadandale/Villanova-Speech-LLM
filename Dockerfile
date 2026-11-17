# Same environment as CONDA_README.md: Python 3.12.3, torch 2.9.1 (cu128) and the same pinned packages.
# The CUDA devel image provides nvcc, which DeepSpeed needs to JIT-compile its ops.
# Ubuntu 24.04 ships Python 3.12.3, the version CONDA_README.md uses.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# A100 (8.0), A40/3090 (8.6), L40S (8.9), H100 (9.0)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    CUDA_MODULE_LOADING=LAZY \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1

# System packages. ffmpeg replaces `conda install -c conda-forge ffmpeg`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        git wget curl ca-certificates build-essential ninja-build \
        ffmpeg sox libsox-fmt-all libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv stands in for the conda env (Ubuntu 24.04 blocks pip installs into the system Python).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
        packaging editdistance gpustat wandb einops debugpy \
        tqdm soundfile matplotlib scipy sentencepiece pandas \
        scikit-learn deepspeed openai-whisper \
        evaluate datasets sacrebleu whisper_normalizer \
        bitsandbytes==0.49.1 accelerate==1.12.0 transformers==4.57.6

# The scripts expect the repo at $WORK/Villanova-Speech-LLM, data at $WORK/data and models at
# $WORK/Villanova-Speech-LLM/models. Mount data and models at run time rather than baking them into the image.
ENV WORK=/workspace
WORKDIR $WORK/Villanova-Speech-LLM

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# As in the training scripts, the code is used from src/ via PYTHONPATH rather than pip-installed
ENV PYTHONPATH=$WORK/Villanova-Speech-LLM/src

# Non-root user; override the IDs to match yours so files written to mounted volumes keep your ownership.
# The ubuntu:24.04 base already has a user with UID/GID 1000, so it is removed first.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd --gid $GROUP_ID app \
    && useradd --uid $USER_ID --gid $GROUP_ID -m app \
    && chown -R app:app $WORK
USER app

CMD ["/bin/bash"]
