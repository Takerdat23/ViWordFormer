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
class CNN_Model_Vipher_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(CNN_Model_Vipher_ABSA, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device= config.device
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        # self.rotary_emb = RotaryEmbedding(dim=self.d_model)
        self.d_model_map = nn.Linear(self.d_model, config.embed_dim)
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.kernel_size =config.kernel_sizes
                             
        self.conv1d_list = nn.ModuleList([
                        nn.Conv1d(in_channels=config.embed_dim,
                                out_channels=num_filters,
                                kernel_size=self.kernel_size[i])
                        for i in range(len(self.kernel_size))
                    ])
        self.dropout = nn.Dropout(config.dropout)
        self.outputHead = Aspect_Based_SA_Output(config.dropout  , len(self.kernel_size) * num_filters, config.output_dim, config.num_categories )
        self.num_labels= config.output_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
      
        x = self.d_model_map(x)
        
        x_reshaped = x.permute(0, 2, 1) # (b, dim, seq)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        
        out = self.dropout(x_fc)
        out = self.outputHead(out)

        # Mask aspects 
        mask = (labels != 0)  
       
        # Filter predictions and labels using the mask
        filtered_out = out.view(-1, self.num_labels)[mask.view(-1)]
        filtered_labels = labels.view(-1)[mask.view(-1)]

        # Compute the loss only for valid aspects
        loss = self.loss(filtered_out, filtered_labels)

        return out, loss
    
    
    
