import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

# @META_ARCHITECTURE.register()
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.dropout(out[:, -1, :])

        # fully connected layer
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # Initialize
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        return h0, c0