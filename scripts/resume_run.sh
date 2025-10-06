#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1

export SBATCH_OUTPUT="logs/$1.out"

source ~/sentence_diffusion/venv/bin/activate

python train.py --wandb_id=$1
