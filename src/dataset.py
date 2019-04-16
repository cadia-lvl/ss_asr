import sys

import torch
import pandas as pd
import numpy as np

from math import ceil

import typing

from torch.utils.data import DataLoader, Dataset

from preprocess import ALL_CHARS, TOKENS, SOS_TKN, EOS_TKN
from postprocess import trim_eos


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
            sort_key:str='', sort_ascending:bool=True):
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
    
    def get_text(self, idx:int) -> str:
        '''
        Returns the text at the given index
        '''
        return self._frame.iloc[idx]['normalized_text']

    def get_batched_texts(self, start_idx:int, pad_token:str=SOS_TKN) -> np.ndarray:
        '''
        Returns a [batch_size, max_len] array of the text for
        the given batch start index in the encoded representation
        where each sample has been padded up to the maximum length
        '''
        encoded = [self.encode(self.get_text(idx)) 
            for idx in self._batch_range(start_idx)]

        lens = [s.shape[0] for s in encoded]
        max_len = max(lens)
        if lens != sorted(lens, reverse=True):
            print(lens)
            t_lens = []
            for idx in self._batch_range(start_idx):
                text = self.get_text(idx)
                print(text)
                t_lens.append(len(text))
                #print(self.encode(self.get_text(idx)))
            print(t_lens)
            print(t_lens == lens)
            sys.exit()

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
        encoded = [self.char2idx(c) for c in text]
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
        '''
        if self.text_only:
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

    def translate(self, seq, return_string=False):
        '''
        Input arguments:
        seq (torch.Tensor): A tensor containing a sequence 
        of indexes
        return_string: If true, a string is returned, rather 
        than a list of characters.
        '''
        new_seq = []
        for c in trim_eos(seq):
            new_seq.append(self.r_mapping[c])

        if return_string:
            new_seq = ''.join(new_seq).replace(SOS_TKN,'').replace(EOS_TKN,'')
        return new_seq

def load_dataset(path: str, batch_size:int=1, n_jobs:int=8, text_only:bool=False, 
    use_gpu:bool=False, sort_key='', sort_ascending=True):
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
        sort_key=sort_key, sort_ascending=sort_ascending)

    return Mapper(), dataset, DataLoader(dataset, batch_size=1, 
        num_workers=n_jobs, pin_memory=use_gpu)

def simple_dataset_test(index_path='data/processed/index.tsv'):
    '''
    Sanity test for a batch size of 1 , add the dataset to the return of load_dataset
    to test
    '''
    dataset, dataloader = load_dataset(index_path, text_only=True)

    for batch_idx, data in enumerate(dataloader):
        for i in range(data.shape[0]):
            print(batch_idx, dataset.decode_from_numpy(np.reshape(data[i,:,:], [data.shape[2]])))

def simple_shape_check(index_path='data/processed/index.tsv'):
    '''
    Just for checking shapes of things
    '''
    dataloader = load_dataset(index_path, batch_size=32)
    _ , data = next(enumerate(dataloader))
    
    print(data[0].shape)
    print(data[1].shape)

if __name__ == '__main__':
    simple_shape_check()