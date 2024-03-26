#!/bin/bash
#SBATCH -J radial_arm_maze
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p long
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH -D /proj/internal_group/dscig/kdkyum/workdir/radial_arm_maze

module load anaconda3/2020.11
module load cudatoolkit/10.2

conda activate pydreamer
export WANDB_MODE=offline
export PYTHONPATH='.'

HYDRA_FULL_ERROR=1 python train_epn_ram.py
