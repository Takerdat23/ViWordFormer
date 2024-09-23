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


class Topic_SA_Output(nn.Module): 
    def __init__(self, d_input, topic_output, sentiment_output):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Topic_SA_Output, self).__init__()
        self.Topicdense = nn.Linear(d_input , topic_output,  bias=True)
        self.SentimentDense = nn.Linear(d_input , sentiment_output,  bias=True)
     
   
    

    def forward(self, model_output ):
        """ 
         x : Model output 
         categories: aspect, categories  
         Output: sentiment output 
        """
        # Mamba (B, L, D)
        pooled_output = model_output[: , 0 , :]

        topic = self.Topicdense(pooled_output )

        sentiment = self.SentimentDense(pooled_output)

        return topic , sentiment

@META_ARCHITECTURE.register()
class MambaClassification(nn.Module):
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
        
        self.lm_head = Topic_SA_Output(self.config.d_model, 3, 4)
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
        
        
        self.loss = nn.CrossEntropyLoss(ignore_index= 0, label_smoothing=0.0)

    def forward(self, tokens, labels):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)
        
        
        x = self.embedding(tokens)

        x = x.reshape(x.size(0), x.size(1), -1)

        x = self.mamba(x)
        x = self.norm_f(x)
        
        

        if self.config.mup:
            x = x / self.config.mup_width_mult

        topic , sentiment = self.lm_head(x)

        labels = labels.squeeze(-1)
        topic_loss = self.loss(topic, labels)
       
   
        return topic , topic_loss , x # return x for testing
    
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
        if isinstance(self.core, Mamba2):
            assert self.config.use_mem_eff_path, "Mamba2 should use the mem_eff_path when decoding with the generate4 function"
            assert prompt.size(1) >= self.config.d_conv

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
    
    """
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult
        
        logits = self.lm_head(x)

        return logits, caches
    
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    """
    
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

# # adapted from https://github.com/johnma2006/mamba-minimal
# def from_pretrained(name: str):
#     """
#     Returns a model loaded with pretrained weights pulled from HuggingFace.

#     Note :
#     This only work with the state-spaces/mamba-XXX model family, because there is a pytorch_model.bin file in the HF repo.
#     This is not the case of typical model saved on HF (like the state-spaces/mamba-XXX-hf model family).
#     To load the state dict of such models, I think the only way is to load the model into a AutoModelForCausalLM, and then
#     pass the state_dict to a MambaLM. I see no other way around unfrortunately (this is how it's done in jamba.py)

#     Args:
#         name: As of now, supports
#             * 'state-spaces/mamba-2.8b-slimpj'
#             * 'state-spaces/mamba-2.8b'
#             * 'state-spaces/mamba-1.4b'
#             * 'state-spaces/mamba-790m'
#             * 'state-spaces/mamba-370m'
#             * 'state-spaces/mamba-130m'

#     Returns:
#         model: a Mamba model configured with the proper parameters and initialized with the proper weights
#     """

#     try:
#         from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
#         from transformers.utils.hub import cached_file
#     except ImportError:
#         print("The from_pretrained function pulls weights from HuggingFace and thus needs transformers to be installed (pip install transformers)")
#         return

#     def load_config_hf(model_name):
#         resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
#         return json.load(open(resolved_archive_file))
                
#     def load_state_dict_hf(model_name):
#         resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
#         return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
#     # copy config data
#     config_data = load_config_hf(name)
#     config = MambaConfig(d_model=config_data['d_model'], n_layers=config_data['n_layers'])
#     model = LM(config, config_data['vocab_size'])

#     # copy weights
#     state_dict = load_state_dict_hf(name)

#     new_state_dict = {}
#     for key in state_dict:
#         if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
#             new_key = key.replace('backbone.', '')
#         else:
#             new_key = key.replace('backbone', 'mamba')

#         new_state_dict[new_key] = state_dict[key]

#     model.load_state_dict(new_state_dict)

#     return model
