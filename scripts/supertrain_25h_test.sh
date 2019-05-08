#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/supertrain.py \
	--name='supertrain_test_25h'\
	--config='./conf/test_malromur2017_25h.yaml'

