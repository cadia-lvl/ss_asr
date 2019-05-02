#!/bin/sh

#SBATCH --mem=10G
#SBATCH --time=1800

~/envs/LAS/bin/python3 src/preprocess.py

