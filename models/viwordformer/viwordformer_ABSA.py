import torch
import torch.nn as nn
import math

from .attention import *
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    '''
        sequences: (bs, seq_len, dim)
    '''

    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_value*__seq.shape[-1])).long() # (b_s, seq_len)
    return mask # (bs, seq_len)

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
    def __init__(self, d_model, max_len=1024):
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

class PhrasalLexemeEncoderLayer(nn.Module):
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int, d_ff: int, dropout: float):
        super().__init__()

        self.head = head
        self.d_q = d_q
        self.d_kv = d_kv

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=head, dropout=dropout, batch_first=True)
        self.phrasal_lexeme_attn = PhrasalLexemeAttention(head, d_model, d_q, d_kv)
        self.linear_out = nn.Linear(head*d_kv, d_model)
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm_1= nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor, phrasal_attn: torch.Tensor):
        '''
            inputs: (bs, nq, d_model)
            attenion_mask: (bs, nq)
            phrasal_attn: (bs, head, nq, nq) - Phrasal Lexeme Attention from previous layers
        '''
        # # performing self-attention
        self_attention_mask = attention_mask * -1e4
        _, self_attn = self.self_attn(inputs, inputs, inputs, self_attention_mask, average_attn_weights=False)

        # performing phrasal lexeme attention
        attention_mask = 1 - attention_mask
        P, phrasal_attn = self.phrasal_lexeme_attn(inputs, attention_mask, phrasal_attn)

        attn_scores = P * self_attn
        b_s, nq = inputs.shape[:2]
        v = self.linear_out(inputs).view(b_s, nq, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nq, d_kv)
        features = torch.matmul(attn_scores, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head * self.d_kv)  # (b_s, nq, h*d_kv)
        
        features = self.norm_1(features + inputs)
        out = self.norm_2(features + self.feed_forward(features))

        return out, self_attn, phrasal_attn, attn_scores

class PhrasalLexemeEncoder(nn.Module):
    def __init__(self, nlayers: int, head: int, d_model: int, d_q: int, d_kv: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            PhrasalLexemeEncoderLayer(head, d_model, d_q, d_kv, d_ff, dropout=dropout)] * nlayers)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        '''
            inputs: (bs, nq, d_model)
            attenion_mask: (bs, nq)
        '''
        # for saving attention scores over layers
        self_attns = []
        phrasal_attns = []
        attn_scores = []

        # At initial the phrasal lexeme attention scores are 0
        phrasal_attn = 0.
        features = inputs
        for layer in self.layers:
            features, self_attn, phrasal_attn, attn_score = layer(features, attention_mask, phrasal_attn)

            self_attns.append(self_attn)
            phrasal_attns.append(phrasal_attn)
            attn_scores.append(attn_score)

        return features, (self_attns, phrasal_attns, attn_scores)

@META_ARCHITECTURE.register()
class ViWordFormer_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.pad_idx = 0
        self.d_model = config.d_model
        self.d_output = config.output_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens,
            embedding_dim=config.d_model
        )
        self.pe = PositionalEncoding(d_model = config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        self.encoder = PhrasalLexemeEncoder(
            nlayers = config.nlayers,
            head = config.head,
            d_model = config.d_model,
            d_q = config.d_q,
            d_kv = config.d_kv,
            d_ff = config.d_ff,
            dropout = config.dropout
        )

        self.outputHead = Aspect_Based_SA_Output(config.dropout , config.hidden_dim, config.output_dim, config.num_categories)
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        padding_mask = generate_padding_mask(input_ids, padding_value=self.pad_idx).to(input_ids.device)

        features = self.embedding(input_ids)
        features = self.pe(features)
        features = self.norm(features)

        features, _ = self.encoder(features, padding_mask)
        out = self.dropout(features[:, 0, :])
        # Fully connected layer
        out = self.outputHead(out)

        # Compute the loss only for valid aspects
        loss = self.loss(out.view(-1, self.d_output), labels.view(-1))

        return out, loss
