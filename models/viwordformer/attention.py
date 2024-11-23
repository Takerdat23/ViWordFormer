import torch
import torch.nn as nn
import torch.nn.functional as F

class PhrasalLexemeAttention(nn.Module):
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int):
        super().__init__()
        
        self.d_model = d_model
        self.head = head
        self.d_q = d_q
        self.d_kv = d_kv

        self.linear_query = nn.Linear(d_model, head * d_q)
        self.linear_key = nn.Linear(d_model, head * d_kv)

    def forward(self, context, attention_mask=None, prior_attn = 0, **kwargs):
        ''''
            context: (bs, seq_len, d_model)
            attention_mask: (bs, seq_len)
            prior_attn: float or (bs, head, seq_len, seq_len)
        '''
        bs, seq_len = context.size()[:2]

        # only pay attention on the afterward token
        after_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32), 1).cuda()
        # only pay attention on the token itself
        self_attention_mask = torch.diag(torch.ones(seq_len , dtype=torch.int32)).cuda()
        # oly pay attention on the previous token
        prev_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32), -1).cuda()
        # for getting P_{ij}
        summing_operator = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32)).cuda()

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask & (after_attention_mask + prev_attention_mask)
        else:
            attention_mask = after_attention_mask + prev_attention_mask
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)

        key = self.linear_key(context).reshape((bs, seq_len, self.head, self.d_q)).permute((0, 2, 1, 3)) # (bs, head, seq_len, d_q)
        query = self.linear_query(context).reshape((bs, seq_len, self.head, self.d_q)).permute((0, 2, 1, 3)) # (bs, head, seq_len, d_q)
        
        phrasal_scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model # (bs, head, seq_len, seq_len)
        
        phrasal_scores = phrasal_scores.masked_fill(attention_mask == 0, -1e9)
        phrasal_scores = F.softmax(phrasal_scores, dim=-1)
        # phrasal attention - attending to "words" in the word-based linguistics view
        phrasal_attn = torch.sqrt(phrasal_scores*phrasal_scores.transpose(-2,-1) + 1e-9)
        # co-text module - forming phrasal lexemes in the morpheme-based linguistics view
        phrasal_attn = prior_attn + (1 - prior_attn)*phrasal_attn

        # First, scatter the diagonal from
        # [
        #   [0, 1, 0, 0],
        #   [0, 0, 2, 0],
        #   [0, 0, 0, 3],
        #   [0, 0, 0, 0],
        # ]
        # to
        # [
        #   [0, 1, 1, 1],
        #   [0, 0, 2, 2],
        #   [0, 0, 0, 3],
        #   [0, 0, 0, 0],
        # ]
        p = torch.log(phrasal_attn + 1e-9).masked_fill(after_attention_mask == 0, 0).matmul(summing_operator)
        # Then, determining the P_{ij}. Note that the matrix P is symmetric, so we only need to keep the upper triangle
        attn = summing_operator.matmul(p).exp().masked_fill((summing_operator.int() - self_attention_mask) == 0, 0)
        # fill up the remaining triangle of the P matrix and perform the residual connection
        attn = attn + attn.transpose(-2, -1) + phrasal_attn.masked_fill(self_attention_mask == 0, 1e-9)
        
        return attn, phrasal_attn
