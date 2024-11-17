import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class Aspect_Based_SA_Output(nn.Module): 
    def __init__(self, dropout , d_input, d_output, num_categories):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Aspect_Based_SA_Output, self).__init__()
        self.dense = nn.Linear(d_input , d_output *num_categories ,  bias=True)
        # self.softmax = nn.Softmax(dim=-1) 
        self.norm = nn.LayerNorm(d_output, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.num_categories = num_categories
        self.num_labels= d_output

    def forward(self, model_output ):
        """ 
         x : Model output 
      
         Output: sentiment output 
        """
       
        x = self.dropout(model_output)
        output = self.dense(x) 
        output = output.view(-1 ,self.num_categories, self.num_labels )
        
        return output

    
@META_ARCHITECTURE.register()
class LSTM_Model_Vipher_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTM_Model_Vipher_ABSA, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device= config.device
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.rotary_emb = RotaryEmbedding(dim=self.d_model)
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.d_model_map = nn.Linear(self.d_model, self.hidden_dim)
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.outputHead = Aspect_Based_SA_Output(config.dropout  , self.hidden_dim, config.output_dim, config.num_categories )
        self.num_labels= config.output_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        x = self.d_model_map(x)

        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)
       
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.dropout(out[:, -1, :])

        # fully connected layer
        out = self.outputHead(out)

        # Mask aspects 
        mask = (labels != 0)  
   
     
        # Filter predictions and labels using the mask
        filtered_out = out.view(-1, self.num_labels)[mask.view(-1)]
        filtered_labels = labels.view(-1)[mask.view(-1)]

        # Compute the loss only for valid aspects
        loss = self.loss(filtered_out, filtered_labels)

        return out, loss
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states and move them to the appropriate device
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0, c0

    
    