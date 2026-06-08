import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# 1. VOCAB & CONSTANTS
char_to_id = {
    "'": 27, "</s>": 2, "<pad>": 0, "<s>": 1, "<unk>": 3,
    "a": 7, "b": 24, "c": 19, "d": 14, "e": 5, "f": 20, "g": 21, "h": 11,
    "i": 10, "j": 29, "k": 26, "l": 15, "m": 17, "n": 9, "o": 8, "p": 23,
    "q": 30, "r": 13, "s": 12, "t": 6, "u": 16, "v": 25, "w": 18, "x": 28,
    "y": 22, "z": 31, "|": 4
}
id_to_char = {v: k for k, v in char_to_id.items()}
VOCAB_SIZE, BLANK_ID = len(char_to_id), 0
        
class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        import whisper
        encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder


class WhisperWrappedCTCEncoder:
    
    @classmethod
    def load(cls, model_config):
        import whisper
        import types
        import torch

        def extract_variable_length_features(self, x: torch.Tensor):
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
            for block in self.blocks:
                x = block(x)
            x = self.ln_post(x)
            return x

        # PATCH: Instead of loading the custom path as a "whisper model",
        # load a standard architecture first to avoid the KeyError: 'dims'
        base_arch = "large-v3-turbo" # TBD 'large-v3' 
        print(f"--> Initializing Whisper architecture: {base_arch}")
        model = whisper.load_model(base_arch, device='cpu')
        encoder = model.encoder

        # Inject the method
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        
        # Note: We don't load the .pth here because WhisperCTCEncoder.__init__ 
        # will call load_state_dict on the whole thing (Encoder + CTC head) 
        # right after this returns.
        
        return encoder
        
class WhisperCTCEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # 1. Load the backbone via the Wrapper (which injects the custom method)
        self.encoder = WhisperWrappedCTCEncoder.load(model_config)
        
        # 2. Define the CTC Head
        self.ctc_head = nn.Linear(model_config.encoder_dim, VOCAB_SIZE)
        self.model_config = model_config

        # 3. LOAD THE FULL CHECKPOINT
        if model_config.encoder_path is not None:
            checkpoint = torch.load(model_config.encoder_path, map_location='cpu')
            state_dict = checkpoint.get("model", checkpoint)
            
            # CRITICAL: Check if we need to add "encoder." prefix to keys
            # If the file has 'conv1.weight' but our param is 'encoder.conv1.weight'
            new_state_dict = {}
            for k, v in state_dict.items():
                # 1. Clean the key of common wrapper prefixes
                clean_key = k.replace("module.", "").replace("model.", "")
                
                # 2. Handle the double 'encoder.' issue
                # If the key is 'encoder.conv1.weight', we want 'encoder.conv1.weight'
                # If the key is just 'conv1.weight', we want 'encoder.conv1.weight'
                
                if clean_key.startswith("ctc_head"):
                    new_state_dict[clean_key] = v
                elif clean_key.startswith("encoder."):
                    # It already has 'encoder.', so it matches self.encoder
                    new_state_dict[clean_key] = v
                else:
                    # It's a raw Whisper key, add 'encoder.' to match self.encoder
                    new_state_dict[f"encoder.{clean_key}"] = v

            msg = self.load_state_dict(new_state_dict, strict=False)
            print(f"--> WhisperCTC Load Result: {msg}")

    @classmethod
    def load(cls, model_config):
        return cls(model_config)

    def forward(self, x):
        # x: [Batch, Mels, Frames]
        # This calls the injected method from WhisperWrappedEncoder
        x = self.encoder.extract_variable_length_features(x)
        logits = self.ctc_head(x)
        
        return {
            "hidden_states": x, 
            "logits": logits
        }
    
class BEATsEncoder:

    @classmethod
    def load(cls, model_config):
        from .BEATs.BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])

        return BEATs_model


@dataclass
class UserDirModule:
    user_dir: str
    
class EATEncoder:
    
    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        EATEncoder = EATEncoder[0]

        return EATEncoder
    
    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask = padding_mask, mask=False, remove_extra_tokens = False)['x']

class CLAPEncoder: 

    @classmethod
    def load(cls, model_config): 
        from .CLAP.ase_model import ASE
        import ruamel.yaml as yaml
        with open(model_config.clap_config, 'r') as f: 
            clap_config = yaml.safe_load(f)
        clap_config['pd_text_support'] = model_config.get("pd_text_support", None)
        model = ASE(clap_config)
        checkpoint = torch.load(model_config.encoder_path)['model']
        model.load_state_dict(checkpoint)
        return model
    
class SpatialASTEncoder:
    @classmethod
    def load(cls, model_config):
        from functools import partial
        from .SpatialAST import SpatialAST 
        binaural_encoder = SpatialAST.BinauralEncoder(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        checkpoint = torch.load(model_config.encoder_ckpt, map_location='cpu')
        binaural_encoder.load_state_dict(checkpoint['model'], strict=False) 
        return binaural_encoder

class WavLMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .wavlm.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"
 
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        return model

class HubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model


class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model

class MusicFMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .musicfm.model.musicfm_25hz import MusicFM25Hz
        model = MusicFM25Hz(
            stat_path = model_config.encoder_stat_path,
            model_path = model_config.encoder_path,
            w2v2_config_path = model_config.get('encoder_config_path', "facebook/wav2vec2-conformer-rope-large-960h-ft")
        )
        return cls(model_config, model)

    def extract_features(self, source, padding_mask=None):
        _, hidden_states = self.model.get_predictions(source)
        out = hidden_states[self.config.encoder_layer_idx]
        return out

class Emotion2vecEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = model[0]

        return model