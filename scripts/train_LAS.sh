#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/train.py \
	--type='asr'\
	--name='malromur2017_default'\
	--config='./conf/malromur2017_default.yaml'

