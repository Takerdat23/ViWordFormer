import torch
import torch.nn as nn
from vocabs.utils.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class RNNmodel_ViPher(nn.Module):
    """
    RNN Model for text classification tasks.
    """
    def __init__(self, config, vocab: Vocab):
        super(RNNmodel_ViPher, self).__init__()
        # Model configuration
        self.device = config.device
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        self.num_layer = config.num_layer
        self.dropout_prob = config.dropout
        self.num_output = config.num_output
        self.bidirectional = config.bidirectional
        self.model_type = config.model_type
        self.label_smoothing = config.label_smoothing

        # Embedding layer
        self.total_token_dict = vocab.total_tokens_dict
        self.embedding_rhyme = nn.Embedding(
            num_embeddings=self.total_token_dict['rhyme'], 
            embedding_dim=self.input_dim, 
            padding_idx=0
        )
        self.embedding_tone = nn.Embedding(
            num_embeddings=self.total_token_dict['tone'], 
            embedding_dim=self.input_dim, 
            padding_idx=0
        )
        self.embedding_onset = nn.Embedding(
            num_embeddings=self.total_token_dict['onset'], 
            embedding_dim=self.input_dim, 
            padding_idx=0
        )

        # linear mapp for rnn
        self.linear_map = nn.Linear(3 * self.input_dim, self.d_model)


        # RNN layer
        if self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(self.d_model * self.bidirectional, self.num_output)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.label_smoothing)

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, 3).
            labels (Tensor): Target labels.

        Returns:
            Tuple: (logits, loss)
        """
        # Split the input into three components
        onset = x[:, :, 0]  # (batch_size, seq_len)
        tone = x[:, :, 1]   # (batch_size, seq_len)
        rhyme = x[:, :, 2]  # (batch_size, seq_len)

        # Embed each feature
        rhyme_embed = self.embedding_rhyme(rhyme)  # (batch_size, seq_len, input_dim)
        tone_embed = self.embedding_tone(tone)    # (batch_size, seq_len, input_dim)
        onset_embed = self.embedding_onset(onset) # (batch_size, seq_len, input_dim)

        # Concatenate embeddings
        x = torch.cat((rhyme_embed, tone_embed, onset_embed), dim=-1)  # (batch_size, seq_len, 3 * input_dim)

        # Apply linear mapping to reduce dimensionality
        x = self.linear_map(x)  # (batch_size, seq_len, d_model)

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

