import random
import torch
from torch.utils.data import DataLoader, Dataset
from preprocess import ALL_CHARS, EOS_TKN, SOS_TKN, TOKENS

class LMDataset(Dataset):
    def __init__(self, filename, chunk_size, chars:str=TOKENS+ALL_CHARS):
        self.file = open(filename).read()
        self.len_file = len(self.file)
        self.chunk_size = chunk_size
        self.chars = chars

        self.char2idx = {chars[i]: i for i in range(len(chars))}
        self.idx2char = {v: k for k, v in self.char2idx.items()}

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

    def s2l(self, s):
        '''
        Input arguments:
        s (str): A string sequence representing an utterance

        Given a sequence of characters, return the label index form 
        tensor version of the string
        '''
        assert type(s) == str
        out = torch.zeros(len(s))
        for i in range(len(s)):
            out[i] = self.char2idx[s[i]]
        return out

    def s2oh(self, s):
        '''
        Input arguments:
        s (str): A string sequence representing an utterance

        Given a sequence of characters, return the one hot 
        tensor version of the string
        '''
        out = torch.zeros(len(s), self.get_num_chars())
        for i in range(len(s)):
            out[i, self.char2idx[s[i]]] = 1
        return out

    def get_num_chars(self):
        return len(self.chars)

    def __len__(self):
        return int(self.len_file / self.chunk_size)

    def __getitem__(self, i):
        '''
        Returns 2 tuples:
        (x_original, y_original), (x, y)
        
        where:
        * x_original, y_original (str): the text input and target
        
        * x, y (tensor): the same, but in indexed label
        form, shapes are [batch_size, chunk_size] where x_l[i,j]
        represents the index of the jth character of the ith sample
        '''
        chunk = self.file[i: i+self.chunk_size+1]
        return (chunk[:-1], chunk[1:]), (self.s2oh(chunk[:-1]).to(self.device),
            self.s2l(chunk[1:]).to(self.device))

def dataload(filename, chunk_size, batch_size, shuffle=True):
    '''
    The dataloader returns batches where the batch dimension comes first,
    i.e. x.shape = [batch_size, chunk_size]
    '''
    ds = LMDataset(filename, chunk_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return ds, dl

def make_split(filename, train_file, eval_file, split=0.9):
    '''
    Splits file at <filename> into train_file (split) and
    val_file (1-split)
    '''
    in_file = open(filename).read()
    train_len = int(split*len(in_file))

    train_txt = in_file[0:train_len]
    eval_txt = in_file[train_len+1:-1]

    with open(train_file, 'w') as t, open(eval_file, 'w') as e:
        t.write(train_txt)
        e.write(eval_txt)

    print("finished splitting")

if __name__ == '__main__':
    ds = LMDataset('./processed_data/risamalromur/clean_total.txt', 200)
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    
    for b_ind, data in enumerate(dl):
        print(b_ind)
    
    '''
    make_split('./processed_data/risamalromur/clean_total.txt', 
        './processed_data/risamalromur/train.txt',
        './processed_data/risamalromur/eval.txt')
    '''