import os
from unittest.mock import MagicMock
import torch.distributed.algorithms.join as join

# THE HACK: Replace the Join class with a Mock that acts as a do-nothing context manager
mock_join = MagicMock()
mock_join.return_value.__enter__.return_value = None
mock_join.return_value.__exit__.return_value = None
join.Join = mock_join

import sys
import datetime
code_dir = "/stek/vskadandale/tok-pruning/SLAM-LLM/examples/asr_librispeech"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
project_root = "/stek/vskadandale/tok-pruning/SLAM-LLM"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = "/stek/vskadandale/tok-pruning/SLAM-LLM/src/"
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

os.chdir(project_root)

from finetune_asr import main_hydra  # Import the function decorated with @hydra.main

# 1. Set Environment Variables (The 'export' section of your bash script)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["PYTHONPATH"] = f"/root/whisper:{os.environ.get('PYTHONPATH', '')}"

# 2. Define your Paths and Variables
date_str = datetime.datetime.now().strftime("%Y%m%d")
llm_path = "/stek/SLAM-LLM/models/eurollm-1.7b-instruct"
train_data_path = "/stek/vskadandale/voxpopuli_jsonl/train.jsonl"
val_data_path = "/stek/vskadandale/voxpopuli_jsonl/val.jsonl"
exp_data = "voxpopuli"
learn_rate = "1e-4"
output_dir = f"/stek/vskadandale/tok-pruning/whisper-sanity-{date_str}"
config_path = f"/stek/vskadandale/tok-pruning/SLAM-LLM/examples/asr_librispeech/conf"

# 3. Construct the Arguments (The 'hydra_args' section)
# We simulate sys.argv as if we ran it from the terminal
sys.argv = [
    "finetune_asr.py",  # Script name (placeholder)
    "--config-path", config_path,
    "--config-name", "prompt.yaml",
    f"hydra.run.dir={output_dir}",
    f"++model_config.llm_name=eurollm-1.7b-instruct",
    f"++model_config.llm_path={llm_path}",
    "++model_config.llm_dim=2048",
    "++model_config.encoder_name=whisper",
    "++model_config.encoder_projector_ds_rate=5",
    "++model_config.encoder_path=large-v3-turbo",
    "++model_config.encoder_dim=1280",
    "++model_config.encoder_projector=linear",
    "++dataset_config.dataset=speech_dataset",
    f"++dataset_config.train_data_path={train_data_path}",
    f"++dataset_config.val_data_path={val_data_path}",
    "++dataset_config.input_type=mel",
    "++dataset_config.mel_size=128",
    "++train_config.model_name=asr",
    "++train_config.num_epochs=10",
    "++train_config.freeze_encoder=true",
    "++train_config.freeze_llm=true",
    "++train_config.batching_strategy=custom",
    "++train_config.warmup_steps=1000",
    "++train_config.total_steps=100000",
    f"++train_config.lr={learn_rate}",
    "++train_config.validation_interval=1000",
    "++train_config.batch_size_training=4",
    "++train_config.val_batch_size=4",
    "++train_config.num_workers_dataloader=2",
    f"++train_config.output_dir={output_dir}",
    "++train_config.use_peft=true",
    "++train_config.peft_config.peft_method=lora",
    "++train_config.enable_fsdp=false",
    "++train_config.enable_ddp=false", # Set to false for easier local debugging
    "++train_config.use_fp16=false",
    "++metric=acc",
    f"++log_config.log_file={output_dir}/train.log",
    "++log_config.use_wandb=false",
    "++debug=False" # This triggers the pdb.set_trace() in your code
]

if __name__ == "__main__":
    # Call the hydra-decorated function directly
    # Hydra will now read from sys.argv
    main_hydra()