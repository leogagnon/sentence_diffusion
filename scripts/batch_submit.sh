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
for arg in "$@"; do
	sbatch scripts/submit.sh sweep_id="$sweep_id" task="$arg"
done




