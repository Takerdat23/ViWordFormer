# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
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

class Attention(nn.Module):
    def __init__(self, hidden_size, da):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(da, da) # Projection for aspect embedding
        self.w = nn.Linear(hidden_size+ da, 1)


    def forward(self, H, va):
        """
        H: (batch_size, seq_len, hidden_size)
        va: (batch_size, da)
        """
        _, seq_len, _ = H.shape
        va_projected = self.Wv(va).unsqueeze(1) # (batch_size, 1, da)
        M_concat = torch.cat((H, va_projected.expand(-1, seq_len, -1)), dim=-1) # (batch_size, seq_len, hidden_size + da)
        
        alpha = self.w(M_concat).squeeze(-1) # (batch_size, seq_len)
        alpha = torch.softmax(alpha, dim=-1) # (batch_size, seq_len)
        r = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) # (batch_size, hidden_size)

        return r


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
        self.num_layer = config.layer_dim
        self.dropout_prob = config.dropout
        self.num_output = config.output_dim
        self.numcategories = config.num_categories
        self.bidirectional = config.bidirectional
        self.model_type = config.model_type
        self.label_smoothing = config.label_smoothing
        self.use_attention = config.use_attention
        self.use_aspect_embedding = config.use_aspect_embedding
        self.num_labels= config.output_dim

        # Embedding layer
        self.pad_idx = 0
        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens,
            embedding_dim=self.input_dim,
            padding_idx=self.pad_idx
        )
         # Aspect embedding layer
        self.da = config.aspect_embedding_dim
        if self.use_aspect_embedding:
            self.aspect_embedding = nn.Embedding(
            num_embeddings=self.numcategories,
            embedding_dim=self.da
            )



        # RNN layer
        if self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_dim if not self.use_aspect_embedding else self.input_dim + self.da,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_dim if not self.use_aspect_embedding else self.input_dim + self.da,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )


        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Attention layer
        if self.use_attention:
            self.attention = Attention(self.d_model * self.bidirectional , self.da) # d_model * bidirectional


        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.label_smoothing)

        self.outputHead = Aspect_Based_SA_Output(self.dropout_prob  , self.d_model*self.bidirectional if not self.use_attention else self.d_model * self.bidirectional, self.num_output, self.numcategories )
    
    def forward(self, x, labels=None, aspects=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
            labels (Tensor): Target labels.
            aspects: Aspect labels (batch_size,)

        Returns:
            Tuple: (logits, loss)
        """
        # Embedding
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        if self.use_aspect_embedding and aspects is not None:
            aspect_emb = self.aspect_embedding(aspects) # (batch_size, num_aspects, da)
            aspect_emb = aspect_emb.mean(dim=1) # (batch_size, da)
            aspect_emb = aspect_emb.unsqueeze(1).expand(-1,x.size(1),-1) # (batch_size, seq_len, da)
            x = torch.cat((x, aspect_emb), dim=-1)


        # Initialize hidden state
        batch_size = x.size(0)
        h0 = self.init_hidden(batch_size)

        # Forward pass
        if 'LSTM' in self.model_type:
            _, (hn, _) = self.rnn(x, (h0, h0)) # hn: (num_layers * num_directions, batch_size, hidden_dim)
        else:
            _, hn = self.rnn(x, h0)

        # Extract the last hidden states from both directions
        idx = -self.bidirectional
        hn = hn[idx:]  # Shape: (2, batch_size, hidden_dim)
        hn = hn.permute(1, 0, 2).reshape(batch_size, -1)  # Shape: (batch_size, 2 * hidden_dim)


        # Dropout and fully connected layer
        out = self.dropout(hn)

        if self.use_attention and aspects is not None:
            aspect_emb = self.aspect_embedding(aspects)
            aspect_emb = aspect_emb.mean(dim=1)
            out = self.attention(hn.view(batch_size, -1, self.d_model * self.bidirectional) , aspect_emb)
        
        out = self.outputHead(out)


        # Compute loss
        if labels is not None:
            batch_size = out.shape[0]
            loss = 0.0
            for i in range(batch_size):
              label = labels[i]
              mask = (label != -1)
              if torch.any(mask): # check if there is any sentiment for this sample, if not then skip, avoid NaN loss
                num_aspect = out.shape[1]
                for j in range(num_aspect):
                    loss += self.loss_fn(out[i, j].view(-1, self.num_labels), label[mask].view(-1))
            loss = loss / batch_size # compute average loss
            return out, loss

        return out

    def init_hidden(self, batch_size):
        """
        Initialize hidden state for the RNN.

        Args:
            batch_size (int): Batch size.

        Returns:
            Tensor: Initialized hidden state tensor.
        """
        return torch.zeros(
            self.num_layer * self.bidirectional,  
            batch_size,
            self.d_model,
            device=self.device
        )