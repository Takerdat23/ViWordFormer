import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class TextCNN(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TextCNN, self).__init__()
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
        self.fc = nn.Linear(len(self.filter_sizes) *
                            self.n_filters, self.output_dim)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        # x shape: (batch size, sentence length)

        # embedded shape: (batch size, sentence length, embedding dim)
        embedded = self.embedding(x)

        # convert to shape: (batch size, 1, sentence length, embedding dim)
        embedded = embedded.unsqueeze(1)
        print(embedded.shape)

        # Convolutions and max-pooling-over-time
        # after conv: (bs, n_filter, seq_len - filter_size + 1, 1) -squeeze(3)
        # -> (bs, n_filter, seq_len - filter_size + 1) ~ (N, C, L)
        # conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #debug by going through each layer
        conved = []
        for conv in self.convs:
            print('Hi conved')
            conved.append(F.relu(conv(embedded)).squeeze(3))

        # [(N, C, L),..] -> [(N, C, 1),..] -> [(N, C),..]
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
        #           for conv in conved]

        #debug by going through each layer
        pooled = []
        for conv in conved:
            print('Hi pooled')
            pooled.append(F.max_pool1d(conv, conv.shape[2]).squeeze(2))
        

        # Concatenate pooled features
        # (N, n_filters * len(filter_sizes))
        cat = self.dropout(torch.cat(pooled, dim=1))

        print('hi cat')
        logits = self.fc(cat)

        print('hi logits')

        if labels is not None:
            loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss

        return logits
