#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:5

~/envs/LAS/bin/python3 src/train.py \
	--type='asr'\
	--name='malromur_default'\
	--config='./conf/malromur2017_default.yaml'

