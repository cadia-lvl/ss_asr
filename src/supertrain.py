#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from solver import SuperSolver

torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training with super solver')
parser.add_argument('--name', type=str, help='Name for logging.', 
    default='supertest')
parser.add_argument('--config', type=str, 
    default='./conf/test_malromur2017_25h.yaml', 
    help='Path to experiment config.')
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

s = SuperSolver(config, paras)
s.exec()