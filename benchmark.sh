#!/bin/bash --login

# Configuración básica para Cirrus
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=mdisspt-s2266011

module load gcc/10.2.0 nvidia/nvhpc/24.5

# Tamaños a probar (potencias de 2)
SIZES=(128 256 512 1024 2048 4096 8192 16384)
POROSITIES=(0.2 0.4 0.6 0.8)

echo "Starting benchmarks at $(date)"
echo "Size,Porosity,Runtime" > results.csv

# Run tests
for SIZE in "${SIZES[@]}"; do
    for POROSITY in "${POROSITIES[@]}"; do
        echo "Testing size ${SIZE}x${SIZE}, porosity ${POROSITY}"
        
        # Run test and save output
        srun -n 1 build/test -M $SIZE -N $SIZE -p $POROSITY -r 3
        
        echo "${SIZE},${POROSITY}" >> results.csv
        echo "----------------------------------------"
    done
done

echo "Benchmarks completed at $(date)" 