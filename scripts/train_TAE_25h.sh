#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/train.py \
	--type='tae'\
	--name='malromur2017_25h'\
	--config='./conf/malromur2017_25h.yaml'

