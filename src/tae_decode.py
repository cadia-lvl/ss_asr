#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from solver import (ADVTrainer, ASRTrainer, SAETrainer,
                    TAETrainer, CHARLMTrainer)

torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--type', type=str, 
    help='asr | tae | sae | adv | test | char_lm',
    default='tae')
parser.add_argument('--name', type=str, help='Name for logging.', 
    default='newtest')
parser.add_argument('--config', type=str, 
    default='./conf/test.yaml', 
    help='Path to experiment config.')
parser.add_argument('--logdir', default='runs/', 
    type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='result/', 
    type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--seed', default=1, type=int, 
    help='Random seed for reproducable results.', required=False)
parser.add_argument('--verbose', default=True, 
    type=bool, required=False)
parser.add_argument('--sanity', default=False)
parser.add_argument('--text', type=str)
parser.add_argument('--drop_rate', type=float, default=0.1)

paras = parser.parse_args()

type_map = {'asr': ASRTrainer, 'tae': TAETrainer, 
    'sae': SAETrainer, 'adv': ADVTrainer, 'char_lm': CHARLMTrainer}

config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

trainer = type_map[paras.type](config,paras)
trainer.load_data()
trainer.set_model()
for drop_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    trainer.encode(paras.text, drop_rate)