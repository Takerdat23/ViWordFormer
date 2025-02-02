import torch
import torch.nn as nn
from .utils import generate_padding_mask, PositionalEncoding
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class TransformerEncoder_ViPher(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TransformerEncoder_ViPher, self).__init__()
        # Model configuration
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_encoder_layers = config.num_encoder_layers
        self.mlp_scaler = config.mlp_scaler
        self.dropout = config.dropout
        self.device = config.device
        self.num_output = config.num_output
        self.label_smoothing = config.label_smoothing
        self.max_seq_len = config.max_seq_len
        self.pad_idx = vocab.get_pad_idx
        self.total_token_dict = vocab.total_tokens_dict

        # Embedding layer
        self.embedding_rhyme = nn.Embedding(
            num_embeddings=self.total_token_dict['rhyme'],
            embedding_dim=self.d_model,
            padding_idx=self.pad_idx
        )
        self.embedding_tone = nn.Embedding(
            num_embeddings=self.total_token_dict['tone'],
            embedding_dim=self.d_model,
            padding_idx=self.pad_idx
        )
        self.embedding_onset = nn.Embedding(
            num_embeddings=self.total_token_dict['onset'],
            embedding_dim=self.d_model,
            padding_idx=self.pad_idx
        )
        # Linear map
        self.linear_map = nn.Linear(3 * self.d_model, self.d_model)

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
            dim_feedforward=self.mlp_scaler * self.d_model,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
            bias=False,
            device=self.device
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=encoder_norm
        )

        # Dense layer
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.num_output)
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        mask = generate_padding_mask(x, self.pad_idx).to(self.device)

        # Split the input into three components
        onset = x[:, :, 0]  # (batch_size, seq_len)
        tone = x[:, :, 1]   # (batch_size, seq_len)
        rhyme = x[:, :, 2]  # (batch_size, seq_len)

        # Embed each feature
        rhyme_embed = self.embedding_rhyme(rhyme)
        tone_embed = self.embedding_tone(tone)
        onset_embed = self.embedding_onset(onset)

        # Concatenate embeddings
        x = torch.cat((rhyme_embed, tone_embed, onset_embed), dim=-1)         # (batch_size, seq_len, 3 * input_dim)

        # Apply linear mapping to reduce dimensionality
        x = self.linear_map(x)  # (batch_size, seq_len, d_model)

        x = self.positional_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = nn.AvgPool1d(kernel_size=x.size(1))(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        logits = self.cls_head(x)

        # Compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss

        return logits


