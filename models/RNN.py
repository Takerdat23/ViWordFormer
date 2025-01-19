import torch
import torch.nn as nn
from vocabs.utils.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class RNNmodel(nn.Module):
    """
    RNN Model for text classification tasks.
    """

    def __init__(self, config, vocab: Vocab):
        super(RNNmodel, self).__init__()
        # Model configuration
        self.device = config.device
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        self.num_layer = config.num_layer
        self.dropout_prob = config.dropout
        self.num_output = config.num_output
        self.bidirectional = config.bidirectional
        self.architecture = config.architecture
        self.label_smoothing = config.label_smoothing
        self.model_type = config.model_type
        # self.tok = config.tok

        # # Linear for vipher
        # self.d_model_map = None
        # if self.tok == "vipher":
        #     self.input_dim = self.input_dim * 3
        #     self.d_model_map = nn.Linear(self.input_dim,
        #                                  self.d_model)

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens,
            embedding_dim=self.input_dim,
            padding_idx=0
        )

        # RNN layer
        _input_size = self.input_dim  # embedding_dim
        _hidden_size = self.d_model
        _num_layers = self.num_layer
        _bidirectional = True if self.bidirectional == 2 else False
        _batch_first = True
        _dropout = self.dropout_prob if self.num_layer > 1 else 0

        _kwargs = {
            'input_size': _input_size,
            'hidden_size': _hidden_size,
            'num_layers': _num_layers,
            'bidirectional': _bidirectional,
            'batch_first': _batch_first,
            'dropout': _dropout,
        }

        if self.model_type == 'GRU':
            self.rnn = nn.GRU(**_kwargs)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(**_kwargs)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(self.d_model * self.bidirectional, self.num_output)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
            labels (Tensor): Target labels.

        Returns:
            Tuple: (logits, loss)
        """
        # Embedding
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # # linear map for Vipher
        # if self.d_model_map:
        #     x = self.d_model_map(x)

        # Initialize hidden state
        batch_size = x.size(0)
        h0 = self.init_hidden(batch_size)

        # Forward pass
        if self.model_type == 'LSTM':
            _, (hn, _) = self.rnn(x, h0)  # Extract hn from the tuple
        else:
            _, hn = self.rnn(x, h0) # hn: (num_layers * num_directions, batch_size, hidden_dim)

        # Extract the last hidden states from both directions
        idx = -self.bidirectional
        hn = hn[idx:]  # Shape: (2, batch_size, hidden_dim)
        hn = hn.permute(1, 0, 2).reshape(batch_size, -1)  # Shape: (batch_size, 2 * hidden_dim)

        # Dropout and fully connected layer
        out = self.dropout(hn)
        logits = self.fc(out)

        # Compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss

        return logits

    def init_hidden(self, batch_size):
        """
        Initialize hidden state for the RNN.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tensor: Initialized hidden state tensor(s).
        """
        if self.model_type == 'LSTM':
            # LSTM requires both h0 (hidden state) and c0 (cell state)
            h0 = torch.zeros(
                self.num_layer * self.bidirectional,
                batch_size,
                self.d_model,
                device=self.device
            )
            c0 = torch.zeros(
                self.num_layer * self.bidirectional,
                batch_size,
                self.d_model,
                device=self.device
            )
            return (h0, c0)
        else:
            # GRU requires only h0 (hidden state)
            return torch.zeros(
                self.num_layer * self.bidirectional,
                batch_size,
                self.d_model,
                device=self.device
            )

