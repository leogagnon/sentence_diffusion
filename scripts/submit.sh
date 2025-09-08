#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=~/sentence_diffusion/logs/misc/%j.out

task_arg=""
for arg in "$@"; do
	if [[ $arg == task=* ]]; then
		task_arg="${arg#task=}"
		break
	fi
done

if [ -z "$task_arg" ]; then
	echo "Error: task=... argument is required."
	exit 1
fi

export SLURM_OUTPUT="logs/${task_arg}.out"
source ~/sentence_diffusion/venv/bin/activate

python train.py "$@"
