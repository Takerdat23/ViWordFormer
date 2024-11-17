import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.viwordformer.attention import ScaledDotProductAttention
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Aspect_Based_SA_Output(nn.Module): 
    def __init__(self, dropout , d_input, d_output, num_categories):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Aspect_Based_SA_Output, self).__init__()
        self.dense = nn.Linear(d_input , d_output *num_categories ,  bias=True)
        # self.softmax = nn.Softmax(dim=-1) 
        self.norm = nn.LayerNorm(d_output, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.num_categories = num_categories
        self.num_labels= d_output

    def forward(self, model_output ):
        """ 
         x : Model output 
      
         Output: sentiment output 
        """
       
        x = self.dropout(model_output)
        output = self.dense(x) 
        output = output.view(-1 ,self.num_categories, self.num_labels )
        
        return output

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

    def forward(self, src, src_mask=None):
        
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

    def forward(self, src, mask):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        return output

@META_ARCHITECTURE.register()
class TransformerModel_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerModel_ABSA, self).__init__()
        self.vocab = vocab
        self.d_model = config.d_model 
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model  , config.dropout)
        encoder_layer = TransformerEncoderLayer(self.d_model, config.head, config.d_ff, config.dropout)
        self.encoder = TransformerEncoder(encoder_layer, config.nlayers)
        self.outputHead = Aspect_Based_SA_Output(config.dropout , config.hidden_dim, config.output_dim, config.num_categories )
        self.dropout = nn.Dropout(config.dropout)
        self.num_labels= config.output_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, src, labels): # src ~ input_id, src_mask ~ attn_mask  
        src_mask = generate_padding_mask(src, padding_value=self.vocab.pad_idx).bool().to(src.device)
        src = self.embedding(src)
   
        src = src.reshape(src.size(0), src.size(1), -1)
        src = self.pos_encoder(src)
      
        output = self.encoder(src, mask=src_mask)
        out = self.dropout(output[:, 0, :])
        # Fully connected layer
        out = self.outputHead(out)
        
        # Mask aspects 
        mask = (labels != 0)  
       
        # Filter predictions and labels using the mask
        filtered_out = out.view(-1, self.num_labels)[mask.view(-1)]
        filtered_labels = labels.view(-1)[mask.view(-1)]

        # Compute the loss only for valid aspects
        loss = self.loss(filtered_out, filtered_labels)

        return out, loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout ,max_len=650):
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