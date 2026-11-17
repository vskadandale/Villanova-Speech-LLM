## GUIDE for setting up CONDA (miniforge3) environment on LEONARDO cineca hpc to work with this repo

Verify CUDA version: module load cuda
Last time I checked it was 12.2

conda create -n vnova python=3.12.3 -y
conda activate vnova

pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

pip install --no-cache-dir \
    packaging editdistance gpustat wandb einops debugpy \
    tqdm soundfile matplotlib scipy sentencepiece pandas \
    scikit-learn deepspeed openai-whisper \
    evaluate datasets sacrebleu whisper_normalizer \
    bitsandbytes==0.49.1 accelerate==1.12.0 transformers==4.57.6

pip install -r $WORK_DIR/Villanova-Speech-LLM/requirements.txt
conda install -c conda-forge ffmpeg -y

cd $WORK_DIR/Villanova-Speech-LLM/
mkdir models
cd models
hf download utter-project/EuroLLM-1.7B-Instruct --local-dir /leonardo_work/IscrC_COBRAS/Villanova-Speech-LLM/models/eurollm-1.7b-instruct
hf download utter-project/EuroLLM-1.7B --local-dir /leonardo_work/IscrC_COBRAS/Villanova-Speech-LLM/models/eurollm-1.7b 
python -c "import torch; from safetensors.torch import save_file; p='$WORK/Villanova-Speech-LLM/models/eurollm-1.7b/pytorch_model.bin'; save_file(torch.load(p, map_location='cpu', weights_only=False), p.replace('pytorch_model.bin', 'model.safetensors'))"
mv $WORK/Villanova-Speech-LLM/models/eurollm-1.7b/pytorch_model.bin $WORK/Villanova-Speech-LLM/models/eurollm-1.7b/pytorch_model.bin.bak

wget https://huggingface.co/SpeechTek/MEUSLI_projector_v2/resolve/main/model.pt
mv model.pt meusli-eurollm-1.7b.pt

wget https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt
mkdir -p ~/.cache/whisper && cp /leonardo_work/IscrC_COBRAS/Villanova-Speech-LLM/models/large-v3-turbo.pt ~/.cache/whisper/
hf download FBK-MT/fama-medium-asr --local-dir /leonardo_work/IscrC_COBRAS/Villanova-Speech-LLM/models/fama-medium-asr
hf download VillanovaAI/Villanova-2B-2603 --local-dir /leonardo_work/IscrC_COBRAS/Villanova-Speech-LLM/models/villanova-2b

cd $WORK_DIR
mkdir data
cd data

rclone copy -P "Personal OneDrive":Datasets/jsonl/voxpopuli_jsonl .
rclone copy -P "Personal OneDrive":Datasets/raw/mosel_vox_populi/en.tar.gz .
tar -xzvf en.tar.gz
rm -rf en.tar.gz
sed -i 's|/stek/corpora|/leonardo_work/IscrC_COBRAS/data|g' *.jsonl

