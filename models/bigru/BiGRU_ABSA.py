import torch
import torch.nn as nn
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
        self.dense = nn.Linear(2*d_input , d_output *num_categories ,  bias=True)
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
class BiGRU_Model_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.device = config.device
        self.d_model = config.d_model
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        
        self.gru = nn.GRU(config.input_dim, self.d_model, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.outputHead = Aspect_Based_SA_Output(config.dropout  , self.hidden_dim, config.output_dim, config.num_categories )
        self.num_labels= config.output_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        
        x = self.embedding(x)
    
        
        batch_size = x.size(0)

        h0 = self.init_hidden(batch_size, self.device)
        _, hn = self.gru(x, h0)
        hn = hn[-2:]
        hn = hn.permute((1, 0, 2)).reshape(batch_size, -1)

        out = self.dropout(hn)

        # Fully connected layer
        out = self.outputHead(out)
        
        loss = self.loss(out.view(-1, self.num_labels), labels.view(-1))
        
        return out, loss
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states
        h0 = torch.zeros(self.layer_dim*2, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0
