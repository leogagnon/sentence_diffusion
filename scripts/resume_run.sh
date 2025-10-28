#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --account=aip-glaj

source ~/sentence_diffusion/venv/bin/activate

srun python train.py --wandb_id=$1
