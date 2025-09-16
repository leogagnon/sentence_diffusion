#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1

source ~/sentence_diffusion/venv/bin/activate

python generate_labels.py
