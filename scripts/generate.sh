#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/generate.py \
    --name='char_lm_main'\
    --config='./conf/malromur2017_default.yaml'