#!/bin/sh

#SBATCH --mem=64G
#SBATCH --time=1800
#SBATCH --gres=gpu:4

~/envs/LAS/bin/python3 src/train.py \
    --type='char_lm'\
    --name='char_lm_final_adadelta'\
    --config='./conf/malromur2017_default.yaml'