import torch
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


    
@META_ARCHITECTURE.register()
class LSTM_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTM_Model, self).__init__()
        self.device= config.device
        self.d_model = config.d_model 
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.d_model, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.d_model, config.output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
  
        x = self.embedding(x)
        
        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)
        
       
        _, (hn, _) = self.lstm(x, (h0, c0))

        out = self.dropout(hn[-1])

        # fully connected layer
        out = self.fc(out)
        
        return out, self.loss(out, labels.squeeze(-1))
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states and move them to the appropriate device
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()

        return h0, c0
    

