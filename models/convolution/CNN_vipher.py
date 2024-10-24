import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE



    
@META_ARCHITECTURE.register()
class CNN_Model_Vipher(nn.Module):
    def __init__(self, config, vocab: Vocab, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(CNN_Model_Vipher, self).__init__()
        NUMBER_OF_COMPONENTS = 3
        self.device= config.device
        self.d_model = config.d_model * NUMBER_OF_COMPONENTS
        self.rotary_emb = RotaryEmbedding(dim=self.d_model)
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.d_model_map = nn.Linear(self.d_model, self.hidden_dim)
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
                             
        self.convs_1d = nn.ModuleList([ nn.Conv2d(1, num_filters, (k, self.d_model), padding=(k-2,0)) for k in kernel_sizes])
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, config.output_dim) 
        self.loss = nn.CrossEntropyLoss()

    
    
    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)
        
        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x, labels):
        
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # x = self.rotary_emb.rotate_queries_or_keys(x)
        
        x = self.d_model_map(x)

        batch_size = x.size(0)

        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        out = self.fc(x) 

        
        return out, self.loss(out, labels.squeeze(-1))
    
    
    
