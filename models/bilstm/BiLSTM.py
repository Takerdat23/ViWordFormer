import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


    
@META_ARCHITECTURE.register()
class BiLSTM_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(BiLSTM_Model, self).__init__()
        self.device= config.device
        self.d_model = config.d_model 
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.d_model, self.layer_dim, bidirectional=True, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.d_model * 2, config.output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
  
        x = self.embedding(x)
        
        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)
        
       
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.dropout(out[:, -1, :])

        # fully connected layer
        out = self.fc(out)
        
        return out, self.loss(out, labels.squeeze(-1))
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states and move them to the appropriate device
        h0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0, c0
    

