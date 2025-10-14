#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4

source ~/sentence_diffusion/venv/bin/activate

srun python train.py --wandb_id=$1
