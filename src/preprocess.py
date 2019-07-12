import os
import re
import argparse
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from tqdm import tqdm

from librosa.core import load, power_to_db
from librosa.feature import melspectrogram
import librosa

CHARS = 'abcdefghijklmnoprstuvxy0123456789'
ICE_CHARS = 'áéíóúýæöþð'
SPECIAL_CHARS = ' .,?'
ALL_CHARS = CHARS + ICE_CHARS + SPECIAL_CHARS
# The SOS will also be used to pad target texts to maximum length
# which is fine since we give the loss function the command to never
# care about this token
SOS_TKN = '<' 
EOS_TKN = '>'
UNK_TKN = '$'
TOKENS = SOS_TKN + EOS_TKN + UNK_TKN

N_JOBS = 12 # no. jobs to run in parallel when writing the index
N_DIMS = 40 # no. frequency coefficients in the spectrograms
WIN_SIZE = 25 # size of window in STFT
STRIDE = 10 # stride of the window
TEXT_XTSN = '.txt' # extension of token files (if applicable)

def preprocess(txt_dir: str, wav_dir: str, processed_dir: str=None):
    if processed_dir is None:
        processed_dir = os.path.join('data', 'processed')
    fbank_dir = os.path.join(processed_dir, 'fbanks')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(fbank_dir, exist_ok=True)

    lines = iterate_by_ids(txt_dir, wav_dir, processed_dir)
    print('Sorting by frame length...')
    lines = sorted(lines, key=lambda x: x[3])
    max_len = lines[-1][3]
    print('Iterating files ...')
    with open(os.path.join(processed_dir, 'index.tsv'), 'w', encoding='utf-8') as f:
        for line in lines:
            '''
            Layout of index
            normalized_text, path_to_fbank, s_len, unpadded_num_frames, text_fname, wav_fname
            '''
            f.write('\t'.join([str(attr) for attr in line]) + '\n')
            
    print('Featurebanks have been computed, now zero-padding (max_len={})...'.format(max_len))
    for line in lines:
        fbank_path = line[1]
        fbank = np.load(fbank_path)
        np.save(fbank_path, zero_pad(fbank, max_len))
    print('Finished zero-padding.')

def iterate_by_ids(txt_dir: str, wav_dir: str,  processed_dir: str):
    '''
    Iterates files in either the txt_dir or wav_dir 
    directory, normalizes the text and preprocesses the audio.
    A preprocessed folder under [root]/data will be created,
    containing the preprocessed data
    '''

    executor = ProcessPoolExecutor(max_workers=N_JOBS)
    futures = []

    for fname in os.listdir(txt_dir):
        if os.path.splitext(fname)[1] == TEXT_XTSN:
            text_path = os.path.join(txt_dir, fname)
            wav_path = os.path.join(wav_dir, os.path.splitext(fname)[0]+'.wav')
            futures.append(executor.submit(partial(process_pair,
                text_path, wav_path, processed_dir)))

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def process_pair(text_path: str, wav_path: str, processed_dir: str):
    # process text
    raw_text = text_from_file(text_path)
    clean_text, s_len = normalize_string(raw_text)

    # process audio
    try:
        sample_rate, y = load_wav(wav_path)
    except:
        print("Error reading wav: {}. Sample is ommitted.".format(wav_path))
        return None
    fbank = log_fbank(y, sample_rate)

    num_frames = fbank.shape[0]

    # save filterbank under <processed_dir>/fbanks/file_id.npy
    fbank_path = os.path.join(processed_dir, 'fbanks', 
        os.path.splitext(os.path.basename(text_path))[0])
    np.save(fbank_path, fbank)
    
    return (clean_text, fbank_path+'.npy', s_len, num_frames, text_path, wav_path)

def preprocess_malromur(index: str, wav_dir: str, processed_dir: str=None):
    '''
    Needs documentation
    '''
    if processed_dir is None:
        processed_dir = os.path.join('data', 'processed')
    fbank_dir = os.path.join(processed_dir, 'fbanks')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(fbank_dir, exist_ok=True)

    lines = iterate_malromur_index(index, wav_dir, processed_dir)
    print('Sorting by frame length...')
    lines = sorted(lines, key=lambda x: x[3])
    max_len = lines[-1][3]
    print('Iterating files ...')
    with open(os.path.join(processed_dir, 'index.tsv'), 'w', encoding='utf-8') as f:
        for line in lines:
            '''
            Layout of index
            normalized_text, path_to_fbank, s_len, unpadded_num_frames, text_fname, wav_fname
            '''
            f.write('\t'.join([str(attr) for attr in line]) + '\n')
            
    print('Spectrograms have been computed, now zero-padding (max_len={})...'.format(max_len))
    for line in lines:
        fbank_path = line[1]
        fbank = np.load(fbank_path)
        np.save(fbank_path, zero_pad(fbank, max_len))
    print('Finished zero-padding.')

def iterate_malromur_index(index_path: str, wav_dir: str, processed_dir: str):
    '''
    The malromur index has the following columns in a comma seperated format
    1) Name of the recording (filename excluding ".wav")
    2) Environment
    3) Some number
    4) Gender of speaker
    5) Age of speaker
    6) Text read
    7) Duration of file in seconds
    8) Classification of recording

    We need to take the name of recording, add '.wav' to it
    and the text read.
    '''
    executor = ProcessPoolExecutor(max_workers=N_JOBS)
    futures = []
    with open(index_path, 'r') as file:
        for line in file:
            line_data = line.rstrip().split(',')
            if line_data[7] == 'correct':
                wav_name = line_data[0] # without extension
                text = line_data[5]
                wav_path = os.path.join(wav_dir, wav_name+'.wav')
                futures.append(executor.submit(partial(process_malromur_pair,
                    text, wav_path, processed_dir)))

    return [future.result() for future in tqdm(futures) if future.result() is not None]

def process_malromur_pair(text: str, wav_path: str, processed_dir: str):
    # process text
    clean_text, s_len = normalize_string(text)

    # process audio
    try:
        sample_rate, y = load_wav(wav_path)
    except:
        print("Error reading wav: {}. Sample is ommitted.".format(wav_path))
        return None
    fbank = log_fbank(y, sample_rate)

    num_frames = fbank.shape[0]

    # save filterbank under <processed_dir>/fbanks/file_id.npy
    fbank_path = os.path.join(processed_dir, 'fbanks', 
        os.path.splitext(os.path.basename(wav_path))[0])
    np.save(fbank_path, fbank)
    
    # we return 'na' as text_path to comply with other stuff
    return (clean_text, fbank_path+'.npy', s_len, num_frames, 'na', wav_path)


def log_fbank(y: np.ndarray, sample_rate:int) -> np.ndarray:
    '''
    Given a signal and a sample rate, calculate the 
    log mel filterbank of the signal.

    Returns a [N_DIMS x num_frames] numpy array
    '''
    ws = int(sample_rate*0.001*WIN_SIZE)
    st = int(sample_rate*0.001*STRIDE)

    fbank = melspectrogram(y, sr=sample_rate, n_mels=N_DIMS, 
    n_fft=ws, hop_length=st)

    # take the log and avoid 0 numerical value issues
    fbank = np.log(fbank + np.finfo(float).eps).astype('float32')
    
    # swap feature and time axis, so each fbank is now
    # [seq, feature], to comply with pytorch

    fbank = np.swapaxes(fbank, 0, 1)

    return fbank

def load_wav(file_path: str) -> Tuple[int, np.ndarray]:
    '''
    reads in a .wav file, returns 
    the sample rate and signal
    '''
    y, sample_rate = load(file_path)
    return sample_rate, y

def text_from_file(file_path: str) -> str:
    '''
    Concats all lines from a file into a single
    string and strips special characters.
    '''
    s = ''
    return ''.join(s for s in open(file_path, 'r')).strip()

def normalize_string(s: str, append_tokens=True) -> Tuple[str, int]:
    '''
    1. Lower case
    2. Alphanumerics (a,b,c,..,0,1,2..)
    6. collapse whitespace
    3. Extra: Space, Comma, Period.
    4. Includes Icelandic special characters 
    (á, ð, é, í, ó, ú, þ, æ, ý, ö)
    5. Other is mapped to <UNK>

    returns the normalized string as well as the string length
    before normalization
    '''
    
    # we have to add 2 because of start ,end token padding 
    s = s.lower()
    s = re.sub(r'\s+', ' ', s) # collapse whitespace
    s_len = len(s) + 2 
    s = re.sub(r"[^0-9{}]".format(CHARS+ICE_CHARS+SPECIAL_CHARS), 
        UNK_TKN, s)

    # pad with <sos> and <eos>
    if append_tokens:
        s = SOS_TKN + s + EOS_TKN

    return s, s_len

def zero_pad(fbank: np.ndarray, max_len: int) -> np.ndarray:
    '''
    Pad a feature bank with zeros on the
    time axis to the maximum length of the 
    whole dataset.

    max_len = 3
    fbank = |a b|
            |c d|

    returns:|a b|
            |c d|
            |0 0| 
    '''
    padded = np.zeros([max_len, N_DIMS])
    padded[:fbank.shape[0], :fbank.shape[1]] = fbank
    return padded

def make_split(index:str, train_r:float=0.9, eval_r:float=0.1):
    '''
    input arguments:
    * index (str): Path to the index generated by the preprocess 
    script
    * train_r (float): The ratio of the dataset designated 
    for training samples
    * eval_r (float): The ratio of the dataset designated for 
    validation samples
    * do_sort (bool): If True, each generated index will be 
    sorted by length. 
    
    Given a preprocessed dataset, and train/eval split, 
    this function splits the dataset and generates 2 different 
    indexes at the same path as the
    original index.
    '''
    assert (train_r + eval_r == 1.0), "Ratios must sum to 1.0"
    
    frame = pd.read_csv(index, sep='\t')
    msk = np.random.rand(len(frame)) < train_r

    train_frame = frame[msk]
    eval_frame = frame[~msk]

    base, _ = os.path.split(index)
    
    train_frame.to_csv(os.path.join(base, 'train.tsv'), sep='\t', index=False)
    eval_frame.to_csv(os.path.join(base, 'eval.tsv'), sep='\t', index=False)

def sort_index(index:str, sort_key:str, sort_ascending:bool=True, out_index:str=None):
    '''
    Input arguments:
    index (str): Path to an index file
    sort_key (str): Key to sort by
    sort_ascending (bool): if True, index is sorted in ascended,
    descended otherwise.

    Given an index generated by the preprocess script, this function
    will sort the index by a column key
    '''
    from ASRDataset import load_df
    frame = load_df(index)
    frame = frame.sort_values(by=[sort_key], ascending=sort_ascending)
    if out_index is not None: index = out_index
    frame.to_csv(index, sep='\t', index=False, header=False)

def subset_by_t(t: float, index: str, out_index: str, avg_utt_s=4.5):
    from dataset import load_df
    '''
    Generate a new index file     
    input arguments:
    * t (int): How large, in seconds, the subset should be
    * index (str): Path to a dataset index
    * out_index (str): Path to where the new index should be stored
    * avg_utt_s (float), default=4.5 : The average utterance length
    in seconds of the dataset used.
    '''
    df = load_df(index)
    num_to_sample =  int(t/avg_utt_s)

    assert num_to_sample < len(df)
    
    sampled_df = df.sample(n=num_to_sample)
    sampled_df.to_csv(out_index, sep='\t', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='dataset', help='Type of dataset')
    malromur_parser = subparsers.add_parser("malromur")
    generic_parser = subparsers.add_parser("generic")
    
    # malromur
    malromur_parser.add_argument('output_dir',
        metavar='o', 
        type=str,
        help='The name of the main output folder under ./data')
    malromur_parser.add_argument('index',
        type=str, 
        help='The path to the malromur index file')
    malromur_parser.add_argument('wav_dir',
        type=str,
        help='The path to the wav directory of Malromur')

    # generic dataset
    generic_parser.add_argument('output_dir',
        metavar='o',
        type=str,
        help='The name of the main output folder under ./data')
    generic_parser.add_argument('wav_dir',
        type=str,
        help='The path to the wav directory of the dataset')
    generic_parser.add_argument('txt_dir', 
        type=str,
        help='The path to the txt directory of the dataset')

    args = parser.parse_args()
    if args.dataset == 'malromur':
        print('Preprocessing Malromur')
        preprocess_malromur(args.index, args.wav_dir, args.o)
    else:
        print('Preprocessing a generic dataset')
        preprocess(args.txt_dir, args.wav_dir, args.o)


