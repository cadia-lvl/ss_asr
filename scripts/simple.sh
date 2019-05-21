#!/bin/sh

#SBATCH --mem=10G
#SBATCH --time=1800
#SBATCH --gres=gpu:0

~/envs/LAS/bin/python3 src/simple_test.py \