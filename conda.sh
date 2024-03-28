#!/bin/bash
#SBATCH -J radial_arm_maze
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p normal
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH -D /proj/internal_group/dscig/kdkyum/workdir/radial_arm_maze

module load anaconda/23.09.0
module load cudatoolkit/11.7

conda config --append envs_dir /proj/internal_group/dscig/kdkyum/workdir/radial_arm_maze

conda env list

conda info --envs
