import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.utils.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class CNN_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(CNN_Model, self).__init__()
        self.device = config.device
        self.d_model = config.d_model
        self.num_filters = config.num_filters
        self.kernel_sizes = config.kernel_sizes  # List of kernel sizes for convolutional layers
        self.output_dim = config.output_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab.total_tokens, self.d_model, padding_idx=0)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.d_model, 
                      out_channels=self.num_filters, 
                      kernel_size=ks) 
            for ks in self.kernel_sizes
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

        # Fully connected layer
        self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.output_dim)

        # Loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        # Embedding layer
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, d_model, seq_len)

        # Apply each convolutional layer and global max pooling
        conv_results = [F.adaptive_max_pool1d(torch.relu(conv(x)), 1).squeeze(-1) for conv in self.convs]
        x = torch.cat(conv_results, dim=1)  # Shape: (batch_size, num_filters * len(kernel_sizes))

        # Dropout
        x = self.dropout(x)

        # Fully connected layer
        out = self.fc(x)

        # Calculate loss
        return out, self.loss(out, labels.squeeze(-1))
