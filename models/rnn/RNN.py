import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class RNNModel(nn.Module):
    def __init__(self, config ,vocab: Vocab  ) -> None:
        super(RNNModel, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device= config.device
        self.d_model = config.d_model  * NUMBER_OF_COMPONENTS
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.rnn = nn.RNN(config.input_dim, self.d_model, config.layer_dim, 
                          batch_first=True, nonlinearity='tanh', 
                          dropout=config.dropout if config.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, 2)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels) -> torch.Tensor:
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        batch_size = x.size(0)
      
        h0 = self.init_hidden(batch_size, self.device)

        out, hn = self.rnn(x.float(), h0.detach())

        out = self.dropout(out[:, -1, :])  # Apply dropout
        out = self.fc(out)
        return out , self.loss(out, labels.squeeze(-1))

    def init_hidden(self, batch_size, device) -> torch.Tensor:
        # Initialize hidden state
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        return h0