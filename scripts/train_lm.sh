#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:2

~/envs/LAS/bin/python3 src/train.py \
	--type='rnn_lm'\
	--name='malromur2017_default'\
	--config='./conf/malromur2017_default.yaml'

