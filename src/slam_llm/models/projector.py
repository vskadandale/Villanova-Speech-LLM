import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random

class EncoderProjectorPruning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.vocab_size = config.ctc_vocab_size
        self.llm_dim = config.llm_dim
        self.encoder_name = config.encoder_name
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.downsampling_type = config.downsampling_type
        if self.k == 1:
            linear_size_out = 2560 #2048 #int(self.encoder_dim / 2)
        else:
            linear_size_out = 2560 #2048
        self.add_layer_norm = config.get("add_layer_norm", False)
        if self.add_layer_norm:
            self.layer_norm = nn.LayerNorm(self.encoder_dim * self.k)
        if self.downsampling_type == "linear":
            if self.vocab_size > 0:
                self.linear1 = nn.Linear(self.vocab_size * self.k, linear_size_out)
            else:
                self.linear1 = nn.Linear(self.encoder_dim * self.k, linear_size_out)
        elif self.downsampling_type == "avgpool":
            self.linear1 = nn.Linear(self.encoder_dim, linear_size_out)
            self.avgpool = nn.AvgPool2d((self.k,1), stride=(self.k,1))
        elif self.downsampling_type == "random":
            self.linear1 = nn.Linear(self.encoder_dim, linear_size_out)
            # We use a fixed seed (e.g., 42) for scientific reproducibility
            self.rng_gen = torch.Generator()
            self.rng_gen.manual_seed(42)
        self.linear2 = nn.Linear(linear_size_out, config.llm_dim)
        self.relu = nn.ReLU()
        self.pad_token = config.encoder_pad
        self.prune_thr = config.encoder_prune_thr
        self.softmax = nn.Softmax(dim=-1)
        self.pruning_type = config.pruning_type
        self.use_dynamic_pruning = config.use_dynamic_pruning
        self.dynamic_prune_thresholds_set = config.dynamic_prune_thresholds_set

    def forward(self, x, y):
        #torch.set_printoptions(profile="full")
        #print("Y:",y)
        tot_frames = x.size(1)
        pruned_frames = 0
        #print("TFrames:",tot_frames, x.size())
        if self.pruning_type == "scalar":  #pruning with the cosine distance
            xo=[]
            #torch.set_printoptions(profile="full")            
            z=torch.roll(x,shifts=-1,dims=1)
            cos_d=self.cos(x,z)
            #print(cos_d.size(), x.size(), z.size(), y.size())
            #print("X:",cos_d[0])
            #print("Y:",y[0,:,0])
            for p in range(0,x.size(0)):
                idx = (cos_d[p] <= self.prune_thr)#.squeeze(1)
                #print("I:",idx)                
                xo += [x[p][idx,:]]
                #print("ZS:",z)
                #z = torch.gather(x[p], dim=0, index=idx)
                #print("ZS:",z)
            x=pad_sequence(xo,batch_first=True)
            pruned_frames = tot_frames - x.size(1)            
            #print("X:",x.size())
        elif self.pruning_type == "ctc":
            y = self.softmax(y)
            l_max,ind_max = torch.topk(y, k =1, dim=-1)
            xo=[]
            for p in range(0,x.size(0)):
                #ind_0 = torch.where(ind_max[p] != self.pad_token)
                if self.use_dynamic_pruning and self.dynamic_prune_thresholds_set:
                    self.prune_thr = random.choice(self.dynamic_prune_thresholds_set)
                ind_0 = torch.where((ind_max[p] != self.pad_token)|( (ind_max[p]== self.pad_token) & (l_max[p] < self.prune_thr) ))            
                xo += [x[p][(ind_0[0]),:]]
            x=pad_sequence(xo,batch_first=True)
            pruned_frames = tot_frames - x.size(1)
        else:
            x = x
            
        print("Pruned_Frames:[",pruned_frames,"]/[",tot_frames,"]")
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k

        if self.encoder_name == "fama": 
            if num_frames_to_discard > 0: 
                x = x[:, :-num_frames_to_discard, :]        
            if self.downsampling_type == "linear":
                seq_len = x.size(1)
                x = x.contiguous().view(batch_size, seq_len // self.k, dim * self.k)
                if self.add_layer_norm:
                    x = self.layer_norm(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)

        else:   
            if num_frames_to_discard > 0:
                y = torch.full((x.size(0),self.k - num_frames_to_discard, x.size(2)), 0.0).to(x.get_device())
                x = torch.cat((x,y),1)

            if self.downsampling_type == "linear":
                seq_len = x.size(1)
                x = x.contiguous()
                #print("X2:",x.size())        
                x = x.view(batch_size, seq_len // self.k, dim * self.k)
                #print("X3:",x.size())
                
                x = self.linear1(x)
                #print("X4:",x.size())
                x = self.relu(x)
                #print("X5:",x.size())
                x = self.linear2(x)
                #print("X6:",x.size())
        
        if self.downsampling_type == "avgpool":
            x = x.unsqueeze(1)
            x = self.avgpool(x)   # [B, 1, 1500, 1280] -> [B, 1, 1500/k, 1280]
            x = x.squeeze(1)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            #print("X7:",x.size())

        elif self.downsampling_type == "random":
            batch_size, tot_len, d_dim = x.size()
            target_len = tot_len // self.k
            # Ensure generator is on the same device as input x
            if self.rng_gen.device != x.device:
                self.rng_gen = torch.Generator(device=x.device)
                self.rng_gen.manual_seed(42)

            # 1. Generate random values for each frame
            rand_vals = torch.rand(batch_size, tot_len, generator=self.rng_gen, device=x.device)
            # 2. Pick top-k and SORT to preserve temporal order
            _, indices = torch.topk(rand_vals, k=target_len, dim=1)
            indices = indices.sort(dim=1).values 
            # 3. Gather frames and pass through projector
            indices = indices.unsqueeze(-1).expand(-1, -1, d_dim)
            x = torch.gather(x, dim=1, index=indices)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
        
        if self.pruning_type == "freezing":  #pruning the ouput of the projector with the cosine distance
            tot_tokens = x.size(1)            
            xo=[]
            z=torch.roll(x,shifts=-1,dims=1)
            cos_d=self.cos(x,z)
            #print(cos_d.size(), x.size(), z.size())
            #print("X0:",cos_d[0])
            for p in range(0,x.size(0)):
                idx = (cos_d[p] <= self.prune_thr)#.squeeze(1)
                #print("I:",idx)                
                xo += [x[p][idx,:]]
                #print("ZS:",z)
                #z = torch.gather(x[p], dim=0, index=idx)
                #print("ZS:",z)
            x=pad_sequence(xo,batch_first=True)
            pruned_tokens = tot_tokens - x.size(1)            
            #print("X1:",x.size())
            print("Pruned_Tokens:[",pruned_tokens,"]/[",tot_tokens,"]")
            
        return x


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj
