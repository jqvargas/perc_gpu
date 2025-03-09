#!/bin/bash --login

# ¡Configuración para la máquina Cirrus! 🚀
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=mdisspt-s2266011

# Cargamos los módulos necesarios, ¡que no se nos olvide! 😅
module load gcc/10.2.0 nvidia/nvhpc/24.5

# Tamaños del problema (potencias de 2 desde 128 hasta 16384)
# ¡Vamos a probar desde chiquitito hasta bestia! 💪
SIZES=(128 256 512 1024 2048 4096 8192 16384)

# Porosidades a probar (desde 0.2 hasta 0.8)
# Probamos diferentes densidades del material, ¡a ver qué pasa! 🧪
POROSITIES=(0.2 0.4 0.6 0.8)

# Archivo de salida para guardar los resultados
# ¡Aquí va todo el cotarro! 
OUTPUT="benchmark_results.txt"
echo "¡Arrancando los benchmarks! 🚀 $(date)" > $OUTPUT
echo "Size,Porosity,CPU_Time,GPU_Time,Speedup" >> $OUTPUT

# ¡A darle caña con los benchmarks! 
for SIZE in "${SIZES[@]}"; do
    for POROSITY in "${POROSITIES[@]}"; do
        echo "🏃 Ejecutando prueba con tamaño ${SIZE}x${SIZE}, porosidad ${POROSITY}"
        
        # Ejecutamos el programa y capturamos la salida
        # ¡A ver qué tal rinde! 🎮
        RESULT=$(srun -n 1 build/test -M $SIZE -N $SIZE -p $POROSITY -r 3 2>&1)
        
        # Extraemos los tiempos (¡la parte interesante!)
        CPU_TIME=$(echo "$RESULT" | grep "CPU.*mean" | awk '{print $7}')
        GPU_TIME=$(echo "$RESULT" | grep "GPU.*mean" | awk '{print $7}')
        
        # Calculamos el speedup (¡cuánto más rápido va la GPU!)
        SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
        
        # Guardamos los resultados
        echo "$SIZE,$POROSITY,$CPU_TIME,$GPU_TIME,$SPEEDUP" >> $OUTPUT
        
        # Mostramos el progreso (¡que se vea el curro!)
        echo "    Tiempo CPU: ${CPU_TIME}s"
        echo "   Tiempo GPU: ${GPU_TIME}s"
        echo "   Speedup: ${SPEEDUP}x"
        echo "----------------------------------------"
    done
done

# finito...
echo "¡Benchmarks completados! 🎊 $(date)" >> $OUTPUT

# Mostramos el resumen final
echo " Los resultados se han guardado en $OUTPUT"
echo " Resumen de resultados:"
echo "----------------------------------------"
cat $OUTPUT 