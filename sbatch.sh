#!/bin/bash

#SBATCH --job-name=aria_train
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="conroy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /coc/flash9/fryan6/code/egohead/EgoVerse
source emimic/bin/activate

hostname
nvidia-smi
pwd
ls

export PYTHONIOENCODING=utf-8
srun -u python -u egomimic/trainHydra.py --config-name=train_zarr_cartesian data=aria model=hpt_bc_flow_aria
    
