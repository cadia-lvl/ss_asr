#!/bin/sh

#SBATCH --mem=16G
#SBATCH --time=1800
#SBATCH --gres=gpu:1

~/envs/LAS/bin/python3 src/asr_test.py

