#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

module load gcc/10.2.0 nvidia/nvhpc/24.5
cmd=(
    srun
    -n 1
    ncu
    -o test-${SLURM_JOB_ID} # save to file
    --section MemoryWorkloadAnalysis
    # --kernel 'regex:your_kernel_name' # allow the use of limits, below
    --launch-skip 1
    --launch-count 10 # collect only ten runs
    build/test # application
)
"${cmd[@]}"
