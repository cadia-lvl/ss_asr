#!/bin/sh

#SBATCH --mem=12G
#SBATCH --gres=gpu:1

~/envs/LAS/bin/python3 src/tests.py

