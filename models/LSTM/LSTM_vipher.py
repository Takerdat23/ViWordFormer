import torch
import torch.nn as nn
from vocabs.utils.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

    
@META_ARCHITECTURE.register()
class LSTM_vipher(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTM_vipher, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device= config.device
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.d_model_map = nn.Linear(self.d_model, self.hidden_dim)
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.hidden_dim, config.output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        x = self.d_model_map(x)

        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)
       
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.dropout(out[:, -1, :])

        # fully connected layer
        out = self.fc(out)
       
        return out, self.loss(out, labels.squeeze(-1))
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states and move them to the appropriate device
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0, c0    

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout ,max_len=512):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
        
#         # Compute the positional encodings once in log space
     
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
       
#         pe = self.pe[:, :x.size(1)] 
#         pe = pe.expand(x.size(0), -1, -1)   
#         x = x + pe
#         return self.dropout(x)