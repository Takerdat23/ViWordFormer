import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.viwordformer.attention import ScaledDotProductAttention
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
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
  
        if src_key_padding_mask is None and src_mask is not None:
            src_key_padding_mask = generate_padding_mask(src_mask)
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1)
        
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask.float())[0]

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
class TransformerModel_vipher(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerModel_vipher, self).__init__()
        self.pad_idx = 0
        NUMBER_OF_COMPONENTS = 3
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.emb_dim = config.d_model
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model  , config.dropout)
        self.d_model_map = nn.Linear(self.d_model , config.hidden_dim)
        encoder_layer = TransformerEncoderLayer(config.hidden_dim, config.head, config.d_ff, config.dropout)
        self.encoder = TransformerEncoder(encoder_layer, config.nlayers)
        self.lm_head = OCD_Output(config.hidden_dim, config.output_dim , config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, src, labels): # src ~ input_id, src_mask ~ attn_mask  
        src_mask = generate_padding_mask(src, 0).bool().to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
   
        src = src.reshape(src.size(0), src.size(1), -1)
        src = self.pos_encoder(src)
        
        src = self.d_model_map(src)
      
        output = self.encoder(src, mask=src_mask)
        label= self.lm_head(output)
        return label, self.loss(label, labels.squeeze(-1))
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout ,max_len=1000):
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