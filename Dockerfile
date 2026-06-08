# 1. Base Image
FROM nvcr.io/nvidia/pytorch:25.01-py3
# This covers A100(8.0), A40/3090(8.6), L40S(8.9), and H100(9.0)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV CUDA_MODULE_LOADING=LAZY

# 2. Arguments
ARG USER=stek
ARG USER_ID=1010
ARG USER_GROUP_ID=1010
ARG DEBIAN_FRONTEND=noninteractive

# STAY AS ROOT for the heavy lifting
USER root

# 3. System dependencies (Stay as root for this)
RUN apt-get update && apt-get install -y \
    wget curl git build-essential ffmpeg \
    sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev \
    vim tmux unzip sudo ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Create the user and the Workspace
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER \
    && mkdir -p /workspace \
    && chown -R $USER:$USER /workspace

# 5. Switch to the user
WORKDIR /workspace

RUN pip install --no-cache-dir \
    packaging editdistance gpustat wandb einops debugpy \
    tqdm soundfile matplotlib scipy sentencepiece pandas \
    scikit-learn deepspeed openai-whisper \
    evaluate datasets sacrebleu whisper_normalizer \
    bitsandbytes==0.49.1 accelerate==1.12.0 transformers==4.57.6

# 7. Clone and install repository
RUN git clone https://github.com/X-LANCE/SLAM-LLM.git \
    && cd SLAM-LLM \
    && pip install --no-cache-dir -e .

# 9. Final Environment and Switch to User
ENV PYTHONNOUSERSITE=1
ENV PIP_USER=false
USER $USER

# 8. Environment setup
ENV SHELL=/bin/bash
ENV PATH="/home/${USER}/.local/bin:${PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.12/dist-packages"

