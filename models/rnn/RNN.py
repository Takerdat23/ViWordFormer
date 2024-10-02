import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


# @META_ARCHITECTURE.register()
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.3) -> None:
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, 
                          batch_first=True, nonlinearity='tanh', 
                          dropout=dropout_prob if layer_dim > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        h0 = self.init_hidden(batch_size, device)

        out, hn = self.rnn(x, h0.detach())

        out = self.dropout(out[:, -1, :])  # Apply dropout
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device) -> torch.Tensor:
        # Initialize hidden state
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        return h0