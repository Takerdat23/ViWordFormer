import torch
import torch.nn as nn
from transformers import AutoModel

from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class OutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.d_model, config.output_dim) 

    def forward(self, hidden_state):
        logits = self.classifier(hidden_state)

        return logits
    
@META_ARCHITECTURE.register()
class Phobert_Sequential_Labeling(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.pad_idx = vocab.pad_idx
        self.d_model = config.d_model
        self.num_labels = vocab.total_labels

        self.norm = nn.LayerNorm(config.d_model)

        self.encoder = AutoModel.from_pretrained(config.pretrained_name)
        if config.freeze_pretrain:
            for para in self.encoder.parameters():
                para.requires_grad = False
        
        self.output_head = OutputHead(config)
    
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(input_ids, padding_value=self.pad_idx).to(input_ids.device)
        attention_mask = 1 - attention_mask
        features = self.encoder(input_ids, attention_mask=attention_mask)
    
        features = features.last_hidden_state
       
        out = self.output_head(features)
        
        loss = self.loss(out.view(-1, self.num_labels), labels.view(-1))

        return out, loss

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