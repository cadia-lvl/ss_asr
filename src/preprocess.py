'''
This script preprocesses a speech-text corpora according to
the specification in the original Listen, Attend & Spell paper.
Given a dataset in some format located under data/[dataset_name],
a new formatted dataset will be created under data/preprocessed.
For now, this script assumes that the text files are stored in seperate
TEXT_XTSN files and audio files stored in seperate .wav files.
Will zero-pad the fbansks at the end and sort by original frame length
'''

'''
    TODO: Currently does not catch IO exceptions and will fail writing the index
    in those cases. Should fix that or concurrently write the index.
'''
import os
import re
from typing import Tuple
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
# TODO: Seems it is standard to omitt the question mark
# so perhaps we should just remove it from the text during
# preprocessing. (The '.' is prolly very helpful for predicting
# when to emmitt the <EOS>)
SPECIAL_CHARS = ' .,?'
ALL_CHARS = CHARS + ICE_CHARS + SPECIAL_CHARS


SOS_TKN = '<' 
# The SOS will also be used to pad target texts to maximum length
# which is fine since we give the loss function the command to never
# care about this token

EOS_TKN = '>'
UNK_TKN = '$'
TOKENS = SOS_TKN + EOS_TKN + UNK_TKN

N_JOBS = 12

N_DIMS = 40
WIN_SIZE = 25
STRIDE = 10
TEXT_XTSN = '.txt'


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


def preprocess_malromur(index: str, wav_dir: str, processed_dir: str=None):
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
    i = 0
    with open(index_path, 'r') as file:
        for line in file:
            i+=1
            if i == 100:
                break
            line_data = line.rstrip().split(',')
            if line_data[7] == 'correct':
                wav_name = line_data[0] # without extension
                text = line_data[0]
                wav_path = os.path.join(wav_dir, wav_name+'.wav')
                futures.append(executor.submit(partial(process_malromur_pair,
                    text, wav_path, processed_dir)))
            else:
                print(line_data[7])

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

def log_fbank(y: np.ndarray, sample_rate: int) -> np.ndarray:
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

def normalize_string(s: str) -> Tuple[str, int]:
    '''
    1. Lower case
    2. Alphanumerics (a,b,c,..,0,1,2..)
    6. collapse whitespace
    3. Extra: Space, Comma, Period.
    4. Includes Icelandic special characters (á, ð, é, í, ó, ú, þ, æ, ý, ö)
    5. Other is mapped to <UNK>

    returns the normalized string as well as the string length
    before normalization
    '''
    
    # we have to add 2 because of start ,end token padding 
    s = s.lower()
    s = re.sub(r'\s+', ' ', s) # collapse whitespace
    s_len = len(s) + 2 
    s = re.sub(r"[^0-9{}]".format(CHARS+ICE_CHARS+SPECIAL_CHARS), UNK_TKN, s)

    # pad with <sos> and <eos>
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
    assert (train_r + eval_r == 1.0), "Ratios must equate 1.0"
    
    frame = pd.read_csv(index, sep='\t')
    msk = np.random.rand(len(frame)) < train_r

    train_frame = frame[msk]
    eval_frame = frame[~msk]

    base, _ = os.path.split(index)
    
    train_frame.to_csv(os.path.join(base, 'train.tsv'), sep='\t', index=False)
    eval_frame.to_csv(os.path.join(base, 'eval.tsv'), sep='\t', index=False)

def clean_index_text(index:str):
    '''
    Given an index generated by the preprocess script, this function
    will re-clean the text targets with the clean_text method. This
    can be handy since often times the tokens used in the ASR change.
    '''

    frame = pd.read_csv(index, names=['normalized_text', 'path_to_fbank', 
        's_len', 'unpadded_num_frames', 'text_fname', 'wav_fname'], sep='\t')    
    frame['normalized_text'] = frame.apply(lambda row: cit_helper(row), axis=1) 
    frame.to_csv(index, sep='\t', index=False)    

def cit_helper(row):
    return normalize_string(text_from_file(row['text_fname']))[0]

def sort_index(index:str, sort_key:str, sort_ascending:bool=True):
    '''
    Input arguments:
    index (str): Path to an index file
    sort_key (str): Key to sort by
    sort_ascending (bool): if True, index is sorted in ascended,
    descended otherwise.

    Given an index generated by the preprocess script, this function
    will sort the index by a column key
    '''
    from dataset import load_df
    frame = load_df(index)
    frame = frame.sort_values(by=[sort_key], ascending=sort_ascending)
    frame.to_csv(index, sep='\t', index=False)

def update_slen(index:str):
    '''
    Sometimes, the calculates cleaned string length is wrong. This
    simple function can be run on an index to make sure that the
    length is equal to the actual length of the normalized strings
    '''
    from dataset import load_df
    frame = load_df(index)
    for i, row in frame.iterrows():
        if len(frame.loc[i, 'normalized_text']) != frame.loc[i, 's_len']:
            frame.loc[i, 's_len'] = len(frame.loc[i, 'normalized_text'])
    frame.to_csv(index, sep='\t', index=False)

if __name__ == '__main__':
    #preprocess('data/ivona_speech_data/ivona_txt', 'data/ivona_speech_data/Kristjan_export', processed_dir='data/ivona_processed')
    #make_split('./data/processed/index.tsv')
    #make_split('./data/ivona_processed/index.tsv')
    #clean_index_text('./data/ivona_processed/index.tsv')
    #sort_index('./data/ivona_processed/index.tsv', 's_len', sort_ascending=False)
    #update_slen('./data/ivona_processed/index.tsv')
    #sort_index('./data/processed/index.tsv', 'unpadded_num_frames', sort_ascending=False)

    preprocess_malromur('/data/malromur2017/info.txt', '/data/malromur2017/correct', './processed_data/malromur2017')