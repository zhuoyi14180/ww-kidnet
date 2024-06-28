#!/bin/bash
#SBATCH --job-name=zzy-dissertation
#SBATCH --partition=gecko
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
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
torchrun --nproc_per_node=3 ./train.py --model=transbts --end_epoch=1000 --dataset=brats_ped_2023 --save_freq=5000
torchrun --nproc_per_node=3 ./train.py --model=unet3d --end_epoch=500 --dataset=brats_ped_2023 --save_freq=200

# torchrun --nproc_per_node=3 ./train.py --model=transbts --end_epoch=1000 --dataset=brats_2019 --save_freq=5000 --seed=1000
# torchrun --nproc_per_node=3 ./train.py --model=unet3d --end_epoch=1000 --dataset=brats_2019 --save_freq=5000 --seed=1000