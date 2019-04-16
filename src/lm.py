import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


def test_simple_lm(index_path='./data/ivona_processed/eval.tsv'):
    '''
    Performs a simple sanity LM test on a single sample, given some
    dataset.
    '''
    (_, dataset, dataloader) = load_dataset(index_path, text_only=True)

    batch_idx, data = next(enumerate(dataloader))
    data = data.long()

    lm = torch.load('result/complete_ivona_lm/rnnlm')
    
    lm_hidden = None

    corrects = 0 
    for i in range(1, data.shape[2]):
        current_char = data[:, :, i]
        if i + 1 < data.shape[2]:
            next_char = data[:, :, i+1]

        lm_hidden, lm_out = lm(current_char, [1], lm_hidden)
        print(F.softmax(lm_out.view(lm_out.shape[2])))
        #print(F.softmax(lm_out))
        prediction = dataset.idx2char(torch.argmax(lm_out).item())
        
        current_char = dataset.idx2char(data[0, 0, i].item())
        if i+1  < data.shape[2]: 
            next_char = dataset.idx2char(data[0, 0, i+1].item())

            print('current: {0}, next: {1}, prediction: {2}'
                .format(current_char, next_char, prediction))

            if next_char == prediction: corrects += 1
    
    print('The model had accuracy of {}%'.format(100*corrects/data.shape[2]))
    
if __name__ == '__main__':
    test_simple_lm()

