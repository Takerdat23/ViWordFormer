import torch
import torch.nn as nn
from attention import *
from torch.nn import GELU
import math

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.proj_dff = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_dmodel = nn.Linear(d_ff, d_model)

    def forward(self, features: torch.Tensor):
        features = self.gelu(self.proj_dff(features))
        features = self.dropout(features)
        features = self.proj_dmodel(features)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, features: torch.Tensor):
        pe = self.pe[:, :features.size(1)] 
        pe = pe.expand(features.size(0), -1, -1)   
        features = features + pe
        return self.dropout(features)

class EncoderOutputLayer(nn.Module): 
    def __init__(self, dropout , d_input, d_output):
        super(EncoderOutputLayer, self).__init__()
        self.dense = nn.Linear(d_input, d_output, bias=True)
        self.norm = nn.LayerNorm(d_output, eps=1e-05)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor):
        features = self.dense(features)
        features = self.norm(features)
        features = self.dropout(features)
        return features

class IntermidiateOutput(nn.Module): 
    def __init__(self,  d_input, vocab_size):
        super(IntermidiateOutput, self).__init__()
        self.dense = nn.Linear(d_input, vocab_size, bias=True)
        self.gelu = GELU()

    def forward(self, x):
        x = self.dense(x)
        x = self.gelu(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int, d_ff: int):
        super(EncoderLayer, self).__init__()
        self.self_attn = ScaledDotProductAttention(head, d_model, d_q, d_kv)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.phrasal_lexeme_attn = PhrasalLexemeAttention(head, d_model, d_q, d_kv)

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
    
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
 
        x = self.sublayer[1](x, self.feed_forward)
    
        return x, group_prob, break_prob

class ConstituentModules(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        
        self.layers = clones(layer, N)

    def forward(self, inputs, mask):
        group_prob = 0.
        break_probs = []
        for layer in self.layers:
            x, group_prob, break_prob = layer(inputs, mask, group_prob)
       
            break_probs.append(break_prob)
        break_probs = torch.stack(break_probs, dim=1)

        return x,  break_probs

class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.output = EncoderOutputLayer(dropout, d_model* 4, d_model)

    def forward(self, inputs, mask):
        break_probs = []
        hidden_states =[]
    
        x = self.word_embed(inputs)
        group_prob = 0.
        for layer in self.layers:
            x,group_prob,break_prob = layer(x, mask,group_prob)
            hidden_states.append(x)
            break_probs.append(break_prob)

        break_probs = torch.stack(break_probs, dim=1)
        return x, hidden_states, break_probs
