#!/bin/bash
#SBATCH -J ram_10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p long
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH -D /proj/internal_group/dscig/kdkyum/workdir/radial_arm_maze

module load anaconda/23.09.0
module load cudatoolkit/11.7

__conda_setup="$('/opt/ibs_lib/apps/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate /proj/internal_group/dscig/kdkyum/workdir/conda_envs/ram

export WANDB_MODE=offline
export PYTHONPATH='.'

HYDRA_FULL_ERROR=1 python train_epn_ram.py
