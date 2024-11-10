import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class SpansDetectOutput(nn.Module):
    def __init__(self,  d_input):
        super(SpansDetectOutput, self).__init__()
        self.span_classifier = nn.Linear(d_input, 2) 

    def forward(self, encoder_output ):
  
      
        last_hidden_state = encoder_output  # shape: (batch_size, sequence_length, d_input)
        
        span_logits = self.span_classifier(last_hidden_state)  # shape: (batch_size, sequence_length, 2)

        return  span_logits

    
@META_ARCHITECTURE.register()
class LSTM_Model_Sequence_label(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTM_Model_Sequence_label, self).__init__()
        self.device= config.device
        self.d_model = config.d_model 
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.d_model, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = SpansDetectOutput(config.d_model)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
  
        x = self.embedding(x)
        
        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)
        
       
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # fully connected layer
        features = out
        logits = self.dropout(self.output_head(features))
        print("Logits" , logits.shape)
        print("labels" , labels.shape)

        
        return logits, self.loss(logits, labels.squeeze(-1))
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states and move them to the appropriate device
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0, c0
    

