import sys
import typing
from math import ceil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from postprocess import trim_eos
from preprocess import ALL_CHARS, EOS_TKN, SOS_TKN, TOKENS

def load_df(path:str):
    '''
    A basic function, creating a single API to get and
    create the index dataframe
    '''
    return pd.read_csv(path, sep='\t',
        names=['normalized_text', 'path_to_fbank', 's_len', 'unpadded_num_frames', 
        'text_fname', 'wav_fname'], 
        dtype={'normalized_text': str, 'path_to_fbank': str,
            's_len': int, 'unpadded_num_frames': int, 'text_fname': str,
            'wav_fname': str})

class ASRDataset(Dataset):
    def __init__(self, tsv_file: str, batch_size:int=32,
            chars:str=TOKENS+ALL_CHARS, text_only:bool=False, 
            sort_key:str='', sort_ascending:bool=True, drop_rate:float=0.0):
        '''
        Input Arguments:
        * tsv_file (string): Path to tsv index
        * batch_size (int): Size of each mini batch (LAS: 32) 
        * chars (str): The list of chars defining the mapping.
        e.g. 'abc' -> {a: 0, b: 1, c: 2}
        * text_only (bool): if True, only text is loaded.
        * sort_key (str): if not None, the internal dataframe 
        will be sorted by the supplied key
        * sort_ascending (bool): if True, dataset is sorted by 
        the key in ascending order, else descending order.
        * drop_rate (float): The probability of dropping a character
        upon retrieval (used as a noise model)

        Expected format of index
        (normalized_text, path_to_fbank, s_len, 
        unpadded_num_frames, text_fname, wav_fname)
        '''
        self.text_only = text_only

        self.char2idx_dict = {chars[i]: i for i in range(len(chars))}
        self.idx2char_dict = {v: k for k, v in self.char2idx_dict.items()}
        self.chars = chars

        self._frame = load_df(tsv_file)

        if sort_key:
            self._frame = self._frame.sort_values(
                by=[sort_key], ascending=sort_ascending)
        
        self._feature_dim = self.get_fbank(0).shape[1]
        self.batch_size = batch_size
        self.num_samples = len(self._frame)

        self.batch_inds = np.arange(0, self.num_samples+1, self.batch_size)

        self.drop_rate = drop_rate

    def char2idx(self, char: str) -> int:
        '''
        Map a character to an index
        '''
        return self.char2idx_dict[char]

    def idx2char(self, idx: int) -> str:
        '''
        Map an index to a character
        '''
        return self.idx2char_dict[idx]

    def get_fbank(self, idx:int) -> np.ndarray:
        '''
        Returns the filterbank at the given index
        '''
        return np.load(self._frame.iloc[idx]['path_to_fbank'])

    def get_fbank_by_path(self, path:str) -> np.ndarray:
        '''
        Returns the filterbank at the given file path
        '''
        return np.load(path)

    def get_batched_fbanks(self, start_idx:int) -> np.ndarray:
        '''
        Returns a numpy array of filterbanks stacked on the 
        first dimension.
        Returns a numpy array of shape [batch_size, seq, feature_dim] 
        '''
        fbanks = [self.get_fbank(idx) 
            for idx in self._batch_range(start_idx)]

        return np.stack(fbanks, axis=0)

    def get_batched_fbanks_by_paths(self, paths:typing.List[str]) -> np.ndarray:
        '''
        Returns a numpy array of filterbanks stacked on the 
        first dimensions
        Returns a numpy array of shape [batch_size, seq, feature_dim] 
        '''
        fbanks = [self.get_fbank_by_path(p) for p in paths]
        return np.stack(fbanks)
    
    def get_text(self, idx:int, drop_rate:float=0.0) -> str:
        '''
        Input arguments:
        * idx (int): An index into the dataframe
        * drop_rate (float): Probability of each character
        in the string being dropped from the sample.
        
        Returns the text at the given index
        '''
        text = self._frame.iloc[idx]['normalized_text']
        if drop_rate > 0:
            dropped_text = ''
            for c in text:
                if (c in [EOS_TKN, SOS_TKN]) or (np.random.rand() > drop_rate):
                    # we don't drop the EOS and SOS tokens
                    dropped_text += c
            return dropped_text
            
        return text

    def get_batched_texts(self, start_idx:int, pad_token:str=SOS_TKN, 
        drop_rate:float=0.0) -> np.ndarray:
        '''
        Input arguments :
        * start_idx (int): The index into the dataframe of the first 
        element in the batch
        * pad_token (str): The token used for padding strings
        Returns a [batch_size, max_len] array of the text for
        the given batch start index in the encoded representation
        where each sample has been padded up to the maximum length
        * drop_rate: The probability of each character being dropped
        '''
        encoded = [self.encode(self.get_text(idx, drop_rate)) 
            for idx in self._batch_range(start_idx)]
        lens = [s.shape[0] for s in encoded]
        max_len = max(lens)        
        out = np.zeros([self.batch_size, max_len]) + self.char2idx(pad_token)
    
        for i, e in enumerate(encoded):
            out[i,:e.shape[0]] = e
        return out

    def _batch_range(self, start_idx:int) -> range:
        '''
        Calculates the range of indexes for the given start index
        of a batch
        '''
        return range(start_idx, self._stop_ind(start_idx))

    def _stop_ind(self, start_idx:int) -> int:
        '''
        Based on the batch size and the start index of the current batch,
        calculate the stop index for the current batch
        '''
        return min(start_idx+self.batch_size, self.num_samples)

    def encode(self, text:str) -> np.ndarray:
        '''
        Fetches the text from the given index, maps
        to character indexes and returns as an ndarray
        '''
        return np.array([self.char2idx(c) for c in text])

    def decode(self, inds) -> str:
        '''
        Input arguments: 
        *inds: A numpy array or a tensor representing the encoded version of a string
        Given a [n] shaped np.ndarray or tensor, the corresponding translated
        string is returned
        '''
        return ''.join(self.idx2char(int(ind)) for ind in inds)
    
    def get_framelength(self, idx: int) -> int:
        '''
        Returns the unpadded framelength at the given index
        '''
        return self._frame.iloc[idx]['unpadded_num_frames']

    def get_feature_dim(self) -> int:
        '''
        Returns the feature dimensionality of the filter banks
        (LAS: 40)
        '''
        return self._feature_dim

    def get_char_dim(self) -> int:
        '''
        Get the number of characters in the mapping,
        helpful information in the LAS speller.
        '''
        return len(self.chars)
    
    def __len__(self):
        return len(self.batch_inds) - 1

    def __getitem__(self, idx):
        '''
        The dataset is not ordered in any particular
        way and file ids in most cases do not correspond
        with any particular ordering.
        
        Returns:
        * (fbanks, text), by default
        * (text), if text_only
        * (text, noisy_text) if text_only and drop_rate > 0
        '''
        if self.text_only:
            if self.drop_rate > 0:
                return (self.get_batched_texts(self.batch_inds[idx]),
                    self.get_batched_texts(self.batch_inds[idx], 
                    drop_rate=self.drop_rate))
            else:
                return self.get_batched_texts(self.batch_inds[idx])
        else:
            return (self.get_batched_fbanks(self.batch_inds[idx]), 
                self.get_batched_texts(self.batch_inds[idx]))

class Mapper():
    '''
    A simple class that can easilly be passed around
    for translating indexes to strings, given the tokens
    '''
    def __init__(self, tokens=TOKENS+ALL_CHARS):
        self.mapping = {tokens[i]: i for i in range(len(tokens))}
        self.r_mapping = {v:k for k,v in self.mapping.items()}

    def get_dim(self):
        return len(self.mapping)

    def translate(self, seq):
        '''
        Input arguments:
        seq (torch.Tensor): A tensor containing a sequence 
        of indexes
        Returns: The decoded string
        '''
        new_seq = []
        for c in trim_eos(seq):
            new_seq.append(self.r_mapping[c])

        new_seq = ''.join(new_seq).replace(SOS_TKN,'').replace(EOS_TKN,'')
        return new_seq

    def ind_to_char(self, ind):
        '''
        Input arguments: 
        * ind (int): Mapping value for some character
        '''
        return self.r_mapping[ind]

    def char_to_ind(self, char):
        return self.mapping[char]

def load_dataset(path: str, batch_size:int=1, n_jobs:int=8, text_only:bool=False, 
    use_gpu:bool=False, sort_key='', sort_ascending=True, drop_rate:float=0.0):
    '''
    Input arguments:
    * path (str) : A full or relative path to a train index 
    generated by the preprocessor.
    * batch_size (int): The size of each batch.
    * n_jobs (int): How many subprocesses to spawn for data 
    loading.
    * text_only (bool): If set to True, only the targets 
    (the text parts) of the ASR data is loaded. This can be 
    used in LM training.
    * use_gpu (bool): If set to True and a CUDA device is 
    available, dataloading might be a bit faster.
    * sort_key (str): if not None, the internal dataframe 
    will be sorted by the supplied key
    * sort_ascending (bool): if True, dataset is sorted by 
    the key in ascending order, else descending order.

    Note: Since we have hacked together batching, the torch 
    dataloader will unsqueeze an extra dimension to both x 
    and y resulting in e.g. shapes like 
    [1, batch_size, seq_len, feature_dim] for x. We deal with 
    this in the solver.
    '''

    dataset = ASRDataset(path, batch_size, text_only=text_only, 
        sort_key=sort_key, sort_ascending=sort_ascending, drop_rate=drop_rate)

    return Mapper(), dataset, DataLoader(dataset, batch_size=1, 
        num_workers=n_jobs, pin_memory=use_gpu)



def prepare_x(x, device=torch.device('cpu')):
    '''
    Input arguments:
    * x: A [1, i, j, k] shaped tensor (optional)
    * device: Device for storing batch (torch.device) 
    
    Returns: 
    x: A [i, j, k] shaped tensor
    x_lens: An i-length list of unpadded f-bank lengths
    '''
    x = x.squeeze(0).to(device=device, dtype=torch.float32)
    # HACK: hard coded the 0 symbol        
    x_lens = np.sum(np.sum(x.cpu().data.numpy(), axis=-1)!=0, axis=-1)
    x_lens = [int(sl) for sl in x_lens]
    
    return x, x_lens


def prepare_y(y, device=torch.device('cpu')):
    '''
    Input arguments:
    * y: A [1, l, m] shaped tensor (optional)
    * device: Device for storing batch (torch.device) 

    Returns:
    y: A [l, m] shaped tensor
    state_len: An l-long list of unpadded text lengths
    '''
    y = y.squeeze(0).to(device=device, dtype=torch.long)
    # HACK: hard coded the 0 symbol
    # BUG: This also has the effect that "<" is never counted
    # a character. and thus the string "<hello>" has a length
    # of 6, and not 7. For that reason, we add a one to the count
    # below.
    y_lens = [int(l) + 1 for l in torch.sum(y!=0, dim=-1)]

    return y, y_lens