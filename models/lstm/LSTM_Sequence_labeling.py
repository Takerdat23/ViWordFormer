import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from typing import Tuple, List

class SpansDetectOutput(nn.Module):
    def __init__(self, d_input: int):
        super(SpansDetectOutput, self).__init__()
        self.span_classifier = nn.Linear(d_input, 2) 

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        span_logits = self.span_classifier(encoder_output)  # shape: (batch_size, sequence_length, 2)
        return span_logits


@META_ARCHITECTURE.register()
class LSTM_Model_Sequence_label(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(LSTM_Model_Sequence_label, self).__init__()
        self.device = config.device
        self.d_model = config.d_model
        self.layer_dim = config.layer_dim
        self.hidden_dim = config.hidden_dim

        # Embedding layer with padding
        self.embedding = nn.Embedding(vocab.total_tokens, config.d_model, padding_idx=0)
        self.lstm = nn.LSTM(config.input_dim, self.d_model, self.layer_dim, batch_first=True, dropout=config.dropout if self.layer_dim > 1 else 0)
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = SpansDetectOutput(config.d_model)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor, word_to_token_map: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM sequence labeling model with BPE token handling.

        Args:
            x (torch.Tensor): Input BPE token IDs (batch_size, sequence_length).
            labels (torch.Tensor): Label tensor for each word (batch_size, word_length).
            word_to_token_map (List[List[int]]): Mapping from each word to its token indices in BPE (batch_size, word_length, token_indices).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and loss value.
        """
        x = self.embedding(x)  # Embedding lookup for input tokens
        batch_size = x.size(0)

        h0, c0 = self.init_hidden(batch_size, self.device)

        # Pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Aggregate embeddings for each word based on word_to_token_map
        pooled_features = self.aggregate_embeddings(out, word_to_token_map)
        logits = self.dropout(self.output_head(pooled_features))

        # Adjust logits and labels for loss calculation
        logits = logits.view(-1, 2)  
        labels = labels.view(-1)     

        loss = self.loss(logits, labels)

        return logits, loss

    def aggregate_embeddings(self, lstm_output: torch.Tensor, word_to_token_map: List[List[List[int]]]) -> torch.Tensor:
        """
        Aggregate embeddings based on the word-to-token map using mean pooling.
        Args:
            lstm_output (torch.Tensor): Output from LSTM layer (batch_size, sequence_length, d_model).
            word_to_token_map (List[List[List[int]]]): Mapping from each word to its sentence-relative token indices.
        Returns:
            torch.Tensor: Aggregated word embeddings (batch_size, max_words, d_model).
        """
        batch_size, seq , d_model = lstm_output.size()
       
        max_words = len(word_to_token_map[0]) #Number of actual words
        pooled_output = torch.zeros((batch_size, max_words, d_model), device=lstm_output.device)

        for i, word_map in enumerate(word_to_token_map):
            for j, token_indices in enumerate(word_map):
                if len(token_indices) > 0:
                   
                    token_embeddings = lstm_output[i, token_indices, :]  # Now indexing within the sentence
                    pooled_output[i, j, :] = torch.mean(token_embeddings, dim=0)

        return pooled_output


    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device).requires_grad_()
        return h0, c0
