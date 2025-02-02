import torch
import math
from torch import nn

def generate_padding_mask(seq, pad_token_id=0):
    if seq.ndim == 3:
        '''(bs, seq_len, 3) -> padding = [[[pad_token_id, pad_token_id, pad_token_id]]]'''
        pad_token = torch.Tensor([pad_token_id, pad_token_id, pad_token_id]).unsqueeze(0).unsqueeze(0).to(seq.device)
        return (seq == pad_token).all(dim=-1)
    return seq == pad_token_id
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
