import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class BiGRU_Vipher(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(BiGRU_Vipher, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device = config.device
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.d_model_map = nn.Linear(self.d_model, self.hidden_dim)
        self.gru = nn.GRU(config.input_dim, self.hidden_dim, self.layer_dim, bidirectional=True, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.hidden_dim * 2, config.output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        

        
        x = self.d_model_map(x)
        
        batch_size = x.size(0)

        h0 = self.init_hidden(batch_size, self.device)
        out, hn = self.gru(x, h0)

        out = self.dropout(out[:, -1, :])

        # Fully connected layer
        out = self.fc(out)
        
        return out, self.loss(out, labels.squeeze(-1))
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states
        h0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0
