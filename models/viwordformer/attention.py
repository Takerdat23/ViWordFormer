import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.viwordformer.utils import *

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_q
        self.d_kv = d_kv
        self.head = head

        self.fc_q = nn.Linear(d_model, head * d_q)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)
        self.fc_o = nn.Linear(head * d_kv, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / torch.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att.masked_fill(attention_mask == 0, -1e9)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_kv)  # (b_s, nq, h*d_kv)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att

class PhrasalLexemeAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int):
        super().__init__()
        
        self.d_model = d_model
        self.head = head
        self.d_q = d_q
        self.d_kv = d_kv

        self.linear_query = nn.Linear(d_q, d_model)
        self.linear_key = nn.Linear(d_kv, d_model)

    def forward(self, context, attention_mask, prior_attn = 0):
        _, seq_len = context.size()[:2]

        # only pay attention on the afterward token
        after_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32), 1).cuda()
        # only pay attention on the token itself
        self_attention_mask = torch.diag(torch.ones(seq_len , dtype=torch.int32)).cuda()
        # oly pay attention on the previous token
        prev_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32), -1).cuda()
        # for getting P_{ij}
        summing_operators = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32)).cuda()

        if attention_mask is not None:
            attention_mask = attention_mask & (after_attention_mask + prev_attention_mask)
        else:
            attention_mask = after_attention_mask + prev_attention_mask

        key = self.linear_key(context)
        query = self.linear_query(context)
        
        phrasal_scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model
        
        phrasal_scores = phrasal_scores.masked_fill(attention_mask == 0, -1e9)
        phrasal_scores = F.softmax(phrasal_scores, dim=-1)
        # phrasal attention - attending to "words" in the word-based linguistics view
        phrasal_attn = torch.sqrt(phrasal_scores*phrasal_scores.transpose(-2,-1) + 1e-9)
        # co-text module - forming phrasal lexemes in the morpheme-based linguistics view
        phrasal_attn = prior_attn + (1 - prior_attn)*phrasal_attn

        p = torch.log(phrasal_attn + 1e-9).masked_fill(after_attention_mask == 0, 0).matmul(summing_operators)
        attn = summing_operators.matmul(p).exp().masked_fill((summing_operators.int() - self_attention_mask) == 0, 0)     
        attn = attn + attn.transpose(-2, -1) + phrasal_attn.masked_fill(self_attention_mask == 0, 1e-9)
        
        return attn, phrasal_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_k: int, d_v: int, attention_module: nn.Module):
        super(MultiHeadAttention, self).__init__()

        self.attention = attention_module(head=head, d_model=d_model, d_k=d_k, d_v=d_v)

        self.dropout = nn.Dropout()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        out, _ = self.attention(queries, keys, values, attention_mask, **kwargs)
        
        # normalization after residual connection
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        return out
