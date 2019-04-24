import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset


class LM(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, out_dim: int, 
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
            hidden_size= hidden_dim,
            dropout=dropout_rate)

        self.out = nn.Linear(hidden_dim, out_dim)

        self.best_perplexity = 1000.0 # lower is better
        self.global_step = 0

    def set_global_step(self, step:int):
        self.global_step = step
    
    def get_global_step(self):
        return self.global_step
    
    def set_best_ppx(self, ppx:float):
        self.best_perplexity = ppx

    def get_best_ppx(self):
        return self.best_perplexity

    def forward(self, x, lens, hidden=None):
        x_emb = self.drop_1(self.emb(x))
        
        packed = nn.utils.rnn.pack_padded_sequence(x_emb, lens, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        outputs = self.out(self.drop_2(outputs))

        return hidden, outputs
