#!/bin/bash

# Usage: ./resume_runs.sh <run_id1> <run_id2> ...

for run_id in "$@"; do
	sbatch scripts/resume_run.sh "$run_id"
done
