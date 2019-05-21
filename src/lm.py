import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LM(nn.Module):
    def __init__(self, out_dim: int, emb_dim: int, hidden_dim: int, 
        num_layers: int, dropout_rate: float):
        '''
        Input arguments:
        * emb_dim: The dimensionality of the word embedding
        * hidden_dim: The hidden dimension of the LSTM unit
        * out_dim: The dimensionality of the alphabet
        * num_layers: Number of LSTM layers
        * dropout_rate: The dropout rate
        '''
        super(LM, self).__init__()

        self.emb = nn.Embedding(out_dim, emb_dim)
        
        self.drop_1 = nn.Dropout(dropout_rate)
        self.drop_2 = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=emb_dim,
            hidden_size=hidden_dim,
            dropout=dropout_rate)

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lens, hidden=None):
        
        x_emb = self.drop_1(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(x_emb, lens, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        outputs = self.out(self.drop_2(outputs))

        return hidden, outputs
