import torch
import torch.nn as nn
from vocabs.vocab import Vocab

from transformers import AutoModel
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


@META_ARCHITECTURE.register()
class Phobert_label(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.pad_idx = 0
        self.d_model = config.d_model

        self.norm = nn.LayerNorm(config.d_model)

        self.encoder = AutoModel.from_pretrained(config.pretrained_name)
        
        
        self.output_head = nn.Linear(
            in_features = config.d_model,
            out_features = config.output_dim
        )
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        padding_mask = generate_padding_mask(input_ids, padding_value=self.pad_idx).to(input_ids.device)

        features = self.encoder(input_ids, padding_mask)
    
        features = features.last_hidden_state
        # the cls token is used for capturing the whole sentence and classification
        features = features[:, 0]
        logits = self.dropout(self.output_head(features))

        return logits, self.loss(logits, labels.squeeze(-1))