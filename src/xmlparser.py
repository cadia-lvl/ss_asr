import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path
from tqdm import tqdm
from preprocess import normalize_string

def prepro_file(in_file, out_file):
    '''
    Uses the prepro for LAS on each line of the given file
    '''
    with open(out_file, 'w') as o, open(in_file, 'r') as i:
        for line in tqdm(i):
            o.write(normalize_string(line, append_tokens=False)[0])

def parse(parent_dir, out_path, reset_file=False):
    with open(out_path, 'w' if reset_file else 'a') as out_file:
        for file_path in tqdm(Path(parent_dir).glob('**/*.xml'),
            total=sum(1 for _ in Path(parent_dir).glob('**/*.xml'))):

            root = ET.parse(str(file_path)).getroot()
            ns = '{http://www.tei-c.org/ns/1.0}'
            sentences = ''
            for i, sentence in enumerate(root.iter(ns+'s')):
                s = '' if i == 0 else ' '
                for j, p in enumerate(sentence):
                    if j != 0 and p.tag == ns+'w':
                        # then we add a space before adding
                        s += ' {}'.format(p.text)
                    else:
                        # no space is added
                        s += '{}'.format(p.text)
                sentences += s
            out_file.write(sentences+'\n')