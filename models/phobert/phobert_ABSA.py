import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import tqdm
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
class Phobert_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.pad_idx = 0
        self.d_model = config.d_model

        self.norm = nn.LayerNorm(config.d_model)

        self.encoder = AutoModel.from_pretrained(config.pretrained)
        
        
        self.output_head =  Aspect_Based_SA_Output(config.dropout  , self.hidden_dim, config.output_dim, config.num_categories )
    
        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        padding_mask = generate_padding_mask(input_ids, padding_value=self.pad_idx).to(input_ids.device)

        features = self.encoder(input_ids, padding_mask)
    
        features = features.last_hidden_state
        # the cls token is used for capturing the whole sentence and classification
        features = features[:, 0]
       
        out = self.outputHead(features)
        
        loss = self.loss(out.view(-1, self.num_labels), labels.view(-1))

        return out, loss

    