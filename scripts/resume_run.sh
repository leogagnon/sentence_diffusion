#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --account=aip-glaj

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISABLE_ADDR2LINE=1

source ~/sentence_diffusion/venv/bin/activate

srun python train.py --wandb_id=$1
