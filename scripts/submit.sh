#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1

export SLURM_OUTPUT="logs/${1}_.out"
source ~/sentence_diffusion/venv/bin/activate

python train.py task=$1
