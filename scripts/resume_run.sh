#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=4

source ~/sentence_diffusion/venv/bin/activate

srun python train.py --wandb_id=$1
