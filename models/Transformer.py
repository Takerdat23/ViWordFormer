import math
import torch
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class TransformerEncoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerEncoder, self).__init__()
        # Model configuration
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_encoder_layers = config.num_encoder_layers
        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.dropout
        self.device = config.device
        self.input_dim = config.input_dim
        self.num_output = config.num_output
        self.label_smoothing = config.label_smoothing
        self.max_seq_len = config.max_seq_len
        self.pad_idx = vocab.get_pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens,
            embedding_dim=self.input_dim,
            padding_idx=self.pad_idx
        )

        # Positional encoder
        self.positional_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
            max_seq_length=self.max_seq_len
        )

        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            device=self.device
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=encoder_norm
        )

        # Dense layer
        self.lm_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.num_output)
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        mask = generate_padding_mask(x, self.pad_idx).to(self.device)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.positional_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        logits = self.lm_head(x[:, 0, :])

        # Compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def generate_padding_mask(seq, pad_token_id=0):
    mask = (seq == pad_token_id)
    return mask
