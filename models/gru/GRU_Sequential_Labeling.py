import torch
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from typing import Tuple

class OutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.span_classifier = nn.Linear(config.d_model, config.output_dim) 

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        span_logits = self.span_classifier(encoder_output)  # shape: (batch_size, sequence_length, 2)
        return span_logits


@META_ARCHITECTURE.register()
class GRU_Sequetiale_Labeling(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.device = config.device
        self.d_model = config.d_model
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        # Embedding layer with padding
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.GRU(config.input_dim, self.d_model, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = OutputHead(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM sequence labeling model with BPE token handling.

        Args:
            x (torch.Tensor): Input BPE token IDs (batch_size, sequence_length).
            labels (torch.Tensor): Label tensor for each word (batch_size, word_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and loss value.
        """
        x = self.embedding(x)  # Embedding lookup for input tokens
        batch_size = x.size(0)

        h0 = self.init_hidden(batch_size, self.device)

        # Pass through LSTM
        out, _ = self.lstm(x, h0)

        # Aggregate embeddings for each word based on word_to_token_map
        logits = self.dropout(self.output_head(out))

        loss = self.loss(logits.view(-1, self.output_dim), labels.view(-1))

        return logits, loss


    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()

        return h0
