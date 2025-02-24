#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

module load gcc/10.2.0 nvidia/nvhpc/24.5

srun -n 1 build/test
