#!/bin/bash
#SBATCH --job-name=zzy-dissertation
#SBATCH --partition=falcon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
## module load CUDA

## Set CUDA_VISIBLE_DEVICES for each task
## export NCCL_DEBUG=INFO
## export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
## module load CUDA/12.2 NCCL/2.20.3_for_CUDA12.2


## Execute your program(s) ##
# torchrun --nproc_per_node=1 ./train.py
python ./predict.py