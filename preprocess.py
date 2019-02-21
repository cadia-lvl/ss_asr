'''
This script preprocesses a speech-text corpora according to
the specification in the original Listen, Attend & Spell paper.

Given a dataset in some format located under data/[dataset_name],
a new formatted dataset will be created under data/preprocessed.

For now, this script assumes that the text files are stored in seperate
TEXT_XTSN files and audio files stored in seperate .wav files.
'''

import os
import re
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

from librosa.core import load, power_to_db
from librosa.feature import melspectrogram

ICE_CHARS = 'áéíóúýæöþð'

N_JOBS = 12

N_DIMS = 40
WIN_SIZE = 25
STRIDE = 10
TEXT_XTSN = '.txt'


def preprocess(txt_dir: str, wav_dir: str, processed_dir: str=None):
    if processed_dir is None:
        processed_dir = os.path.join('data', 'processed')
    fbank_dir = os.path.join('data', 'processed', 'fbanks')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(fbank_dir, exist_ok=True)

    print('Iterating files ...')
    lines = iterate_by_ids(txt_dir, wav_dir, processed_dir)

    with open(os.path.join(processed_dir, 'index.tsv'), 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('\t'.join([str(attr) for attr in line]) + '\n')
            '''
            Layout of index
            normalized_text, path_to_fbank, s_len, num_frames, text_fname, wav_fname
            '''
    print('Preprocessing finished')


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
    sample_rate, y = load_wav(wav_path)
    fbank = log_fbank(y, sample_rate)

    num_frames = fbank.shape[1]

    # save filterbank under <processed_dir>/fbanks/file_id.npy
    fbank_path = os.path.join(processed_dir, 'fbanks', 
        os.path.splitext(os.path.basename(text_path))[0])
    np.save(fbank_path, fbank)

    return (clean_text, fbank_path, s_len, num_frames, text_path, wav_path)


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
    return np.log(fbank + np.finfo(float).eps).astype('float32')

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
    s_len = len(s)
    s = s.lower()
    s = re.sub(r'\s+', ' ', s) # collapse whitespace
    s = re.sub(r"[^a-z0-9{}., ]".format(ICE_CHARS), '<UNK>', s)

    return s, s_len

if __name__ == '__main__':
    preprocess('data/ivona_speech_data/ivona_txt', 'data/ivona_speech_data/Kristjan_export')
