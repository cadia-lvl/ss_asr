#!/usr/bin/env python
# coding: utf-8
import os
import sys
import yaml
import torch
import random
import argparse
import numpy as np

from solver import ASR_Trainer


torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')

parser.add_argument('--config', type=str, 
    default='./conf/asr_confs/test.yaml', 
    help='Path to experiment config.')
parser.add_argument('--name', type=str, help='Name for logging.',
    default='las_test')
parser.add_argument('--logdir', default='log/', 
    type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='result/', 
    type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--load', default=None, 
    type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=1, type=int, 
    help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=1, type=int, 
    help='Number of threads for decoding.', required=False)
parser.add_argument('--verbose', default=True, 
    type=bool, required=False)

paras = parser.parse_args()

# check if not loading and using same name
if paras.name is not None and paras.load is None:
    # user has given a certain name, but not indicated
    # that a model is to be loaded. Check if /results/name/rnnlm
    # exists and exit if True
    if os.path.isfile(os.path.join(paras.ckpdir, paras.name, 'asr')):
        print('A pretrained ASR was found at: "{}". Use the load parameter to advance'
            .format(os.path.join(paras.ckpdir, paras.name)))
        sys.exit()

config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

trainer = ASR_Trainer(config,paras)
trainer.load_data()
trainer.set_model()
trainer.exec()
