#!/usr/bin/env bash

cd $HF_HOME/datasets
# Use find + parallel cp --parents to mirror the structure under DEST.
find HuggingFaceFW___fineweb/ -type f | parallel -j 8 --bar cp --parents {} "$SLURM_TMPDIR"

# Set HF_DATASETS_CACHE to the location of the copied dataset in the tmpdir
export HF_DATASETS_CACHE="$SLURM_TMPDIR"

echo "Copied FineWeb to SLURM_TMPDIR and set HF_DATASETS_CACHE to point to it."
