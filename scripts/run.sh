#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100l:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=long

source ~/sentence_diffusion/venv/bin/activate

srun python train.py task/ae=sonar
