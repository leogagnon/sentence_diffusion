#!/bin/bash
# This script submits each argument as a separate sbatch job, with a leading --sweep_id=... argument
sweep_id_arg="$1"
if [[ "$sweep_id_arg" == id=* ]]; then
	sweep_id="${sweep_id_arg#id=}"
else
	echo "Error: First argument must be in the form id=..."
	exit 1
fi
shift
for task in "$@"; do
	sbatch -J "$task" -o "logs/${sweep_id}_${task}.log" scripts/submit.sh task="$task" sweep_id="$sweep_id" 
done




