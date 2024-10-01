"""

Universal language model, which accepts as its core a Mamba or Mamba-2 model.
It has an embedding layer, and a LM head which maps the model output to logits.

"""

from typing import Union, List
import inspect
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from builders.model_builder import META_ARCHITECTURE
from models.mambaModels.mamba1 import Mamba, RMSNorm
from .mambaConfig import MambaConfig


class NLI_Output(nn.Module): 
    def __init__(self, dropout , d_input, Num_labels):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(NLI_Output, self).__init__()
        self.dense = nn.Linear(d_input ,Num_labels ,  bias=True)
        # self.softmax = nn.Softmax(dim=-1) 
        self.norm = nn.LayerNorm(d_input, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.num_labels= Num_labels

    def forward(self, encoder_output ):
        """ 
         x : Model output 
         categories: aspect, categories  
         Output: sentiment output 
        """

        x = encoder_output[:, 0, :]
      
        x= self.norm(x)
        x = self.dropout(x)
        output = self.dense(x)
     
        return output


@META_ARCHITECTURE.register()
class MambaNLI(nn.Module):
    def __init__(self, model_config, vocab, num_of_components = 3 , pad_vocab_size_multiple: int = None):
        super().__init__()
        self.vocab_size = vocab.total_tokens

        if pad_vocab_size_multiple != None and (self.vocab_size % pad_vocab_size_multiple != 0):
            self.vocab_size += (pad_vocab_size_multiple - self.vocab_size % pad_vocab_size_multiple)
    
        self.config = MambaConfig(d_model=model_config.d_model * num_of_components  , n_layers=model_config.n_layers)

        self.embedding = nn.Embedding(self.vocab_size, model_config.d_model, padding_idx=0)
       
        # if isinstance(self.config):
        #     self.mamba = Mamba(self.config)

        self.mamba = Mamba(self.config)
        

        self.norm_f = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)
        
        self.lm_head = NLI_Output(model_config.dropout, self.config.d_model, 5)
        # self.embedding.weight = self.lm_head.weight # weight-tying disabled

        # muP custom initialization
        if self.config.mup and isinstance(self.config):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight']): # # "hidden weights"
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers) # scale down std of layers which projects onto the residual stream (not muP related)

                    if 'mixer.dt_proj.weight' in pn:
                        std = self.config.dt_rank**-0.5 * self.config.dt_scale
                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D']):
                    # keep Mamba default init for these params
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)
        
        elif self.config.mup and isinstance(self.config):
            for pn, p in self.named_parameters():
                
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight']): # # "hidden weights"
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers) # scale down std of layers which projects onto the residual stream (not muP related)

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))               
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D', 'mixer.dt_bias']):
                    # keep Mamba default init for these params
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)

        else:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('mixer.out_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std/math.sqrt(2 * self.config.n_layers))
        
        
        # self.loss = nn.CrossEntropyLoss(ignore_index= 0, label_smoothing=0.0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens, labels):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)
        
        x = self.embedding(tokens)
        

        x = x.reshape(x.size(0), x.size(1), -1)

        x = self.mamba(x)
        x = self.norm_f(x)
        
        

        if self.config.mup:
            x = x / self.config.mup_width_mult

        nli = self.lm_head(x)
        
        # print("logits" , topic)
        # print("labels ", labels)

        labels = labels.squeeze(-1)
        
        loss = self.loss(nli, labels)
       
   
        return nli , loss , x # return x for testing
    
    def generate(self, prompt, num_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt : (B, L)

        # generation : (B, l)

        # L>>l

        if top_k is not None:
            top_k = min(top_k, self.vocab_size)
        
        input_device = prompt.device
        prompt = prompt.to(self.embedding.weight.device)

        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(generated) # (B, L, vocab_size)
                next_token_logits = logits[:, -1]

                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    
                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        self.train()
        
        return generated.to(input_device)[:, -num_tokens:]
    
    def generate4(self, prompt, num_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt : (1, L)

        assert not isinstance(self.core, Mamba), "Mamba1 doesn't support decoding with the generate4 function."
   

        if top_k is not None:
            top_k = min(top_k, self.vocab_size)

        input_device = prompt.device
        model_device = self.embedding.weight.device

        prompt = prompt.to(model_device)

        self.eval()

        len_prompt = prompt.size(1)
        generated = prompt.clone()

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [layer.get_empty_cache(prompt.size(0)) for layer in self.core.layers]

        with torch.no_grad():
            # process prompt in one go
            logits, caches = self.forward(prompt, caches) # (B, L, vocab_size)
            next_token_logits = logits[:, -1] # (B, vocab_size)

            for t in range(num_tokens):
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)

                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True) # (B, 1)

                generated = torch.cat([generated, next_token], dim=1)
                next_token_logits, caches = self.forward(generated[:, [len_prompt+t]], caches, seq_pos=len_prompt+t) # (B, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)

        self.train()
        return generated.to(input_device)[:, -num_tokens:]
    
   
    # non-muP init (taken from llama2.c)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)

    # adaped from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup and isinstance(self.config):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        elif self.config.mup and isinstance(self.config):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D and A

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]
        
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)

        return optimizer
