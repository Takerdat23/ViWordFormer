import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.viwordformer.attention import ScaledDotProductAttention
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerModel, self).__init__()
        self.pad_idx = vocab.pad_idx
        NUMBER_OF_COMPONENTS = 3
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
        encoder_layer = TransformerEncoderLayer(config.d_model, config.head, config.d_ff, config.dropout)
        self.encoder = TransformerEncoder(encoder_layer, config.nlayers)
        self.d_model = config.d_model
        self.decoder = nn.Linear(config.d_model, vocab.total_tokens) # self.decoder ~~ self.output_head 
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=config.label_smoothing)

    def forward(self, src, labels): # src ~ input_id, src_mask ~ attn_mask  
        src_mask = generate_padding_mask(src, self.pad_idx).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, src_key_padding_mask=src_mask)
        output = self.decoder(output[:, 0, :])
        return output, self.loss(output, labels.squeeze(-1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    
    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences
    print(f"Sequence shape: {sequences.shape}")
    print(f"__seq shape: {__seq.shape}")
    print(f"torch.sum(__seq, dim=-1): {torch.sum(__seq, dim=-1)}")
    print(f"padding_value * __seq.shape[-1]: {padding_value * __seq.shape[-1]}")
    print((torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])))
    mask = (torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])).to(torch.long) # (b_s, seq_len)
    return mask # (bs, seq_len)