#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

import trainer

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('type',
    metavar='t',
    type=str,
    nargs='?',
    help='The type of training/testing to perform',
    choices=['ASRTrainer', 'ASRTester', 'LMTrainer', 'TAETrainer', 
        'SAETrainer', 'AdvTrainer'],
    default='ASRTrainer')
parser.add_argument('name',
    metavar='n',
    type=str,
    nargs='?',
    help='Name for logging', 
    default='experiment_1')
parser.add_argument('config',
    metavar='c',
    type=str,
    nargs='?',
    help='Path to experiment config.',
    default='./conf/test.yaml')
parser.add_argument('logdir',
    type=str,
    nargs='?',
    help='Logging path.',
    default='runs/')
parser.add_argument('ckpdir',
    type=str,
    nargs='?',
    help='Checkpoint/Result path.',
    default='result/')
parser.add_argument('--seed',
    type=int,
    help='Random generator seed.',
    default=1)
parser.add_argument('--verbose',
    type=bool,
    help='If set to true, a lot of information is printed (recommended)',
    default=True)

paras = parser.parse_args()
config = yaml.safe_load(open(paras.config,'r'))

# Set the seed of all generators to the selected seed
# for deterministic results
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)
torch.backends.cudnn.deterministic = True

# Setup the selected trainer
sel_trainer = getattr(trainer, paras.type)(config, paras) 
sel_trainer.load_data()
sel_trainer.set_model()
sel_trainer.exec()
