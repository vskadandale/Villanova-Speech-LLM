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
code_dir = "/stek/vskadandale/Villanova-Speech-LLM/examples/asr_fama"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
project_root = "/stek/vskadandale/Villanova-Speech-LLM"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = "/stek/vskadandale/Villanova-Speech-LLM/src/"
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
train_data_path = "/stek/vskadandale/data/voxpopuli_jsonl/train.jsonl"
val_data_path = "/stek/vskadandale/data/voxpopuli_jsonl/val.jsonl"
exp_data="voxpopuli"
dataset_type="fama_dataset"
add_layer_norm=True
learn_rate = "1e-4"
encoder_dim=1024
input_type="mel"
mel_size=80
ds_rate=2
encoder_arch="fama"
speech_encoder_path="/stek/vskadandale/Villanova-Speech-LLM/models/fama-medium-asr"
encoder_projector="pruning"
downsampling_type="linear"
llm_name="villanova-2b"
llm_path = f"/stek/vskadandale/Villanova-Speech-LLM/models/{llm_name}"
llm_dim=2560

input_type="mel"
output_dir = f"/stek/vskadandale/Villanova-Speech-LLM/SANITY-{date_str}"
config_path = f"/stek/vskadandale/Villanova-Speech-LLM/examples/asr_fama/conf"

# 3. Construct the Arguments (The 'hydra_args' section)
# We simulate sys.argv as if we ran it from the terminal
sys.argv = [
    "finetune_asr.py",  # Script name (placeholder)
    "--config-path", config_path,
    "--config-name", "prompt.yaml",
    f"hydra.run.dir={output_dir}",
    f"++model_config.llm_name={llm_name}",
    f"++model_config.llm_path={llm_path}",
    f"++model_config.llm_dim={llm_dim}",
    f"++model_config.encoder_name={encoder_arch}",
    f"++model_config.encoder_projector_ds_rate={ds_rate}",
    f"++model_config.encoder_path={speech_encoder_path}",
    f"++model_config.encoder_dim={encoder_dim}",
    f"++model_config.add_layer_norm={add_layer_norm}",
    f"++model_config.encoder_projector={encoder_projector}",
    f"++model_config.downsampling_type={downsampling_type}",
    f"++dataset_config.dataset={dataset_type}",
    f"++dataset_config.train_data_path={train_data_path}",
    f"++dataset_config.val_data_path={val_data_path}",
    f"++dataset_config.input_type={input_type}",
    f"++dataset_config.mel_size={mel_size}",
    "++train_config.model_name=asr",
    "++train_config.num_epochs=1",
    "++train_config.freeze_encoder=false",
    "++train_config.freeze_llm=true",
    "++train_config.batching_strategy=custom",
    "++train_config.warmup_steps=1000",
    "++train_config.total_steps=100000",
    f"++train_config.lr={learn_rate}",
    "++train_config.validation_interval=1000",
    "++train_config.batch_size_training=4",
    "++train_config.val_batch_size=4",
    "++train_config.num_workers_dataloader=1",
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