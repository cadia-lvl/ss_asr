#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from preprocess import SOS_TKN
from solver import CHARLMTrainer

torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--name', type=str, help='Name for logging.', 
    default='newtest')
parser.add_argument('--config', type=str, 
    default='./conf/test.yaml', 
    help='Path to experiment config.')
parser.add_argument('--start', type=str, default='pétur helgi hefur aldrei ',
    help='The start of the generated string')
parser.add_argument('--length', type=int, default=300,
    help='The total length of the predicted string')
parser.add_argument('--temp', type=float, default=0.6,
    help='Low=more correct, high=more varying')
parser.add_argument('--logdir', default='runs/', 
    type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='result/', 
    type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--seed', default=1, type=int, 
    help='Random seed for reproducable results.', required=False)
parser.add_argument('--verbose', default=True, 
    type=bool, required=False)


paras = parser.parse_args()
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

trainer = CHARLMTrainer(config,paras)
trainer.load_data()
trainer.set_model()
generated = trainer.generate(length=paras.length, temp=paras.temp, start=paras.start)
print(generated)
#trainer.exec()
