#!/bin/sh

#SBATCH --mem=2G
#SBATCH --time=1800
#SBATCH --gres=gpu:1

~/envs/LAS/bin/python3 gpu_test.py
