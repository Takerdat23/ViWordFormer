import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


class OCD_Output(nn.Module): 
    def __init__(self, d_input, label_output, dropout):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(OCD_Output, self).__init__()
        self.labeldense = nn.Linear(d_input , label_output,  bias=True)
     
        self.norm = nn.LayerNorm(d_input, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, model_output ):
        """ 
         x : Model output 
         categories: aspect, categories  
         Output: sentiment output 
        """
        # Mamba (B, L, D)
        pooled_output = model_output[: , 0 , :]
        
        x= self.norm(pooled_output)
        x = self.dropout(x)

        label= self.labeldense(x)


        return label

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]

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

    def forward(self, src, mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        return output

@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.d_model = config.d_model 
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model  , config.dropout)
        encoder_layer = TransformerEncoderLayer(self.d_model, config.head, config.d_ff, config.dropout)
        self.encoder = TransformerEncoder(encoder_layer, config.nlayers)
        self.lm_head = OCD_Output(self.d_model, config.output_dim,  config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, labels): # src ~ input_id, src_mask ~ attn_mask  
        src_mask = generate_padding_mask(src, padding_value=self.vocab.pad_idx).bool().to(src.device)
        src = self.embedding(src)
   
        src = src.reshape(src.size(0), src.size(1), -1)
        src = self.pos_encoder(src)
      
        output = self.encoder(src, mask=src_mask)
        label= self.lm_head(output)

        return label, self.loss(label, labels.squeeze(-1))

    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout ,max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space
     
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
       
        pe = self.pe[:, :x.size(1)] 
        pe = pe.expand(x.size(0), -1, -1)
        x = x + pe
        return self.dropout(x)


def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    
    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences
    
    mask = (torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])).to(torch.long) # (b_s, seq_len)
    return mask # (bs, seq_len)
