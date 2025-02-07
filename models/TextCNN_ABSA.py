import torch
import torch.nn as nn
import torch.nn.functional as F
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
class TextCNN_ABSA(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TextCNN_ABSA, self).__init__()
        # Model configuration
        self.device = config.device
        self.vocab_size = vocab.total_tokens
        self.d_model = config.embedding_dim
        self.n_filters = config.n_filters
        self.filter_sizes = config.filter_sizes
        self.output_dim = config.num_output
        self.dropout = config.dropout
        self.label_smoothing = config.label_smoothing
        self.pad_idx = vocab.get_pad_idx
        
        self.num_categories = config.num_categories

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=self.pad_idx,
        )

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.n_filters,
                      kernel_size=(fs, self.d_model))
            for fs in self.filter_sizes
        ])
        
        # Others layer
        self.dropout = nn.Dropout(self.dropout)
        
        self.cls_head = Aspect_Based_SA_Output(config.dropout , len(self.filter_sizes) *
                            self.n_filters , self.output_dim, self.num_categories )



        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        # x shape: (batch size, sentence length)

        # embedded shape: (batch size, sentence length, embedding dim)
        embedded = self.embedding(x)

        # convert to shape: (batch size, 1, sentence length, embedding dim)
        embedded = embedded.unsqueeze(1)

        # Convolutions and max-pooling-over-time
        # after conv: (bs, n_filter, seq_len - filter_size + 1, 1) -squeeze(3)
        # -> (bs, n_filter, seq_len - filter_size + 1) ~ (N, C, L)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        # [(N, C, L),..] -> [(N, C, 1),..] -> [(N, C),..]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        

        # Concatenate pooled features
        # (N, n_filters * len(filter_sizes))
        cat = self.dropout(torch.cat(pooled, dim=1))

        logits = self.cls_head(cat)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.output_dim), labels.view(-1))
            return logits, loss

        return logits
