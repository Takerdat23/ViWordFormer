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
        self.numcategories = config.num_categories
        self.bidirectional = config.bidirectional
        self.model_type = config.model_type
        self.label_smoothing = config.label_smoothing

        # Embedding layer
        self.pad_idx = vocab.get_pad_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens, 
            embedding_dim=self.input_dim, 
            padding_idx=self.pad_idx
        )

        # RNN layer
        if self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # Fully connected layer
        # self.fc = nn.Linear(self.d_model * self.bidirectional, self.num_output)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.label_smoothing)

        self.outputHead = Aspect_Based_SA_Output(self.dropout_prob  , self.d_model, self.num_output, self.numcategories )

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
        out = self.outputHead(out)
        # logits = self.fc(out)

        # Compute loss
        # if labels is not None:
        #     loss = self.loss_fn(logits, labels.squeeze(-1))
        #     return logits, loss

        # return logits

        if labels is not None:
            loss = self.loss_fn(out.view(-1, self.num_labels), labels.view(-1))
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
