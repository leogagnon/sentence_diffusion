#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1

source ~/sentence_diffusion/venv/bin/activate

python train.py --wandb_it=$1
