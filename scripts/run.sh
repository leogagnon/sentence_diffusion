#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=long-cpu

source ~/sentence_diffusion/venv/bin/activate

python filter_wiki_paragraphs.py
