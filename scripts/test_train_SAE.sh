#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/train.py \
    --type='sae'\
	--name='sae_test'\
	--config='./conf/test_malromur2017_25h.yaml'

