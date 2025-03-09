// -*- mode: C++; -*-
//
// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#include "perc_gpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// ¡Optimización para la GPU V100! 
// Tío, la V100 es una bestia con 80 SMs y puede manejar hasta 2048 threads por SM
// Usamos bloques de 32x32 = 1024 threads que es lo más guay para esta GPU
// [GPU ADAPTATION] - Increased from CPU's serial processing to 32x32 thread blocks
constexpr int BLOCK_SIZE = 32;  // Tamaño del bloque (32x32 threads, ¡a tope!)
constexpr int printfreq = 100;

// Macro para checkear errores de CUDA
// Porsi las moscas, mejor prevenir que curar ;)
#define CHECK_CUDA_ERROR(val) check_cuda( (val), #val, __FILE__, __LINE__ )
template<typename T>
void check_cuda(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n",
                file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

// ¡El kernel más chulo de CUDA para la percolación!
// [GPU ADAPTATION] - Converted from CPU's sequential loop to parallel CUDA kernel
__global__ void percolate_kernel(int M, int N, const int* __restrict__ state, 
                                int* __restrict__ next, int* changes) {
    // Calculamos los índices con el nuevo tamaño de bloque
    // ¡Trucazo! Usamos registros para acceso más rápido
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // [GPU ADAPTATION] - Changed row-major to column-major for coalesced access
    const int j = blockIdx.x * blockDim.x + tx + 1;  // columna (acceso coalescido, ¡qué pro!)
    const int i = blockIdx.y * blockDim.y + ty + 1;  // fila
    
    if (i <= M && j <= N) {
        // Pre-calculamos el índice y lo guardamos en registro
        // ¡Optimización a tope! 🚀
        const int idx = i * (N + 2) + j;
        
        // Cachear el valor actual en registro
        const int oldval = state[idx];
        
        if (oldval != 0) {
            // Guardamos los vecinos en registros
            // [GPU ADAPTATION] - Cache neighbor values in registers for faster access
            const int up = state[idx - (N + 2)];     // Vecino de arriba
            const int down = state[idx + (N + 2)];   // Vecino de abajo
            const int left = state[idx - 1];         // Vecino de la izquierda
            const int right = state[idx + 1];        // Vecino de la derecha
            
            // Usamos registros para los cálculos intermedios
            // ¡Más rápido que un Fórmula 1! 🏎️
            int newval = oldval;
            newval = max(newval, up);
            newval = max(newval, down);
            newval = max(newval, left);
            newval = max(newval, right);

            // Escribimos el resultado (escritura coalescida)
            next[idx] = newval;
            
            // Solo hacemos la operación atómica si hay cambio
            // [GPU ADAPTATION] - Added atomic operation for parallel counting
            if (newval != oldval) {
                atomicAdd(changes, 1);  // ¡Cuenta atómica, que no se nos escape ninguno!
            }
        } else {
            next[idx] = 0;
        }
    }
}

// Estructura para manejar todo el cotarro de la GPU
struct GpuRunner::Impl {
    // Dimensiones de la matriz
    int M;
    int N;
    // [GPU ADAPTATION] - Changed from regular to pinned memory for faster transfers
    int* state;      // Memoria del host (pinned, ¡más rápida que un rayo!)
    int* tmp;        // Memoria del host (pinned)
    int* d_state;    // Memoria de la GPU
    int* d_tmp;      // Memoria de la GPU
    int* d_changes;  // Contador de cambios en la GPU
    int* h_changes;  // Contador de cambios en el host (pinned)

    // Eventos para medir tiempos (¡a cronometrar todo!)
    cudaEvent_t start_compute;
    cudaEvent_t stop_compute;
    cudaEvent_t start_h2d;
    cudaEvent_t stop_h2d;
    cudaEvent_t start_d2h;
    cudaEvent_t stop_d2h;

    // Resultados de tiempos (en milisegundos)
    float compute_time;
    float h2d_time;
    float d2h_time;

    // Streams de CUDA (¡para hacer varias cosas a la vez!)
    // [GPU ADAPTATION] - Added streams for concurrent operations
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    Impl(int m, int n) : M(m), N(n), compute_time(0), h2d_time(0), d2h_time(0) {
        // Allocate pinned memory for all host buffers
        CHECK_CUDA_ERROR(cudaHostAlloc(&state, size() * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&tmp, size() * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_changes, sizeof(int), cudaHostAllocDefault));

        // Initialize host memory to zero
        std::memset(state, 0, size() * sizeof(int));
        std::memset(tmp, 0, size() * sizeof(int));
        *h_changes = 0;

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_state, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(int)));

        // Initialize device memory to zero
        CHECK_CUDA_ERROR(cudaMemset(d_state, 0, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemset(d_tmp, 0, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemset(d_changes, 0, sizeof(int)));

        // Create CUDA events
        CHECK_CUDA_ERROR(cudaEventCreate(&start_compute));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_compute));
        CHECK_CUDA_ERROR(cudaEventCreate(&start_h2d));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_h2d));
        CHECK_CUDA_ERROR(cudaEventCreate(&start_d2h));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_d2h));

        // Create CUDA streams
        CHECK_CUDA_ERROR(cudaStreamCreate(&compute_stream));
        CHECK_CUDA_ERROR(cudaStreamCreate(&transfer_stream));
    }
    
    ~Impl() {
        // Free pinned memory
        if (state) CHECK_CUDA_ERROR(cudaFreeHost(state));
        if (tmp) CHECK_CUDA_ERROR(cudaFreeHost(tmp));
        if (h_changes) CHECK_CUDA_ERROR(cudaFreeHost(h_changes));
        
        // Free device memory
        if (d_state) CHECK_CUDA_ERROR(cudaFree(d_state));
        if (d_tmp) CHECK_CUDA_ERROR(cudaFree(d_tmp));
        if (d_changes) CHECK_CUDA_ERROR(cudaFree(d_changes));

        // Destroy CUDA events
        cudaEventDestroy(start_compute);
        cudaEventDestroy(stop_compute);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);

        // Destroy CUDA streams
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
    }

    int size() const {
        return (M + 2) * (N + 2);
    }

    void print_timing_stats() const {
        printf("\nPerformance Summary:\n");
        printf("Computation time    : %.3f ms\n", compute_time);
        printf("Host to Device time : %.3f ms\n", h2d_time);
        printf("Device to Host time : %.3f ms\n", d2h_time);
        printf("Total time         : %.3f ms\n", compute_time + h2d_time + d2h_time);
        printf("Memory transfer time: %.3f ms (%.1f%%)\n", 
               h2d_time + d2h_time, 
               100.0f * (h2d_time + d2h_time) / (compute_time + h2d_time + d2h_time));
    }
};

GpuRunner::GpuRunner(int M, int N) : m_impl(std::make_unique<Impl>(M, N)) {
}

GpuRunner::~GpuRunner() = default;

void GpuRunner::copy_in(int const* source) {
    // Start timing H2D transfer
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->start_h2d, m_impl->transfer_stream));
    
    // Copy to pinned memory
    std::memcpy(m_impl->state, source, m_impl->size() * sizeof(int));
    
    // Copy to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->d_state, m_impl->state, 
                                    m_impl->size() * sizeof(int), 
                                    cudaMemcpyHostToDevice,
                                    m_impl->transfer_stream));
    
    // Ensure the transfer is complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->transfer_stream));
    
    // Record H2D transfer end and calculate time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->stop_h2d, m_impl->transfer_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->stop_h2d));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, m_impl->start_h2d, m_impl->stop_h2d));
    m_impl->h2d_time += milliseconds;
}

void GpuRunner::copy_out(int* dest) const {
    // Start timing D2H transfer
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->start_d2h, m_impl->transfer_stream));
    
    // Copy from device to pinned memory asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->state, m_impl->d_state,
                                    m_impl->size() * sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    m_impl->transfer_stream));
    
    // Ensure the transfer is complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->transfer_stream));
    
    // Copy from pinned memory to destination
    std::memcpy(dest, m_impl->state, m_impl->size() * sizeof(int));
    
    // Record D2H transfer end and calculate time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->stop_d2h, m_impl->transfer_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->stop_d2h));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, m_impl->start_d2h, m_impl->stop_d2h));
    m_impl->d2h_time += milliseconds;
}

void GpuRunner::run() {
    int const M = m_impl->M;
    int const N = m_impl->N;
    
    // Calculate grid dimensions for 32x32 blocks
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // Now 32x32 threads per block
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Print occupancy information
    printf("\nGrid Configuration:\n");
    printf("Block dimensions : %dx%d threads (%d threads per block)\n", 
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE);
    printf("Grid dimensions  : %dx%d blocks\n", gridDim.x, gridDim.y);
    printf("Total threads    : %d\n", 
           gridDim.x * gridDim.y * BLOCK_SIZE * BLOCK_SIZE);
    printf("Problem size    : %dx%d cells\n", M, N);

    int const maxstep = 4 * std::max(M, N);
    int step = 1;
    int nchange = 1;

    // Use pointers to device buffers
    int* d_current = m_impl->d_state;
    int* d_next = m_impl->d_tmp;

    // Start timing computation
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->start_compute, m_impl->compute_stream));

    while (nchange && step <= maxstep) {
        // Reset change counter
        CHECK_CUDA_ERROR(cudaMemsetAsync(m_impl->d_changes, 0, sizeof(int), m_impl->compute_stream));
        
        // Launch kernel with 32x32 blocks
        percolate_kernel<<<gridDim, blockDim, 0, m_impl->compute_stream>>>(M, N, d_current, d_next, m_impl->d_changes);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Get number of changes
        CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->h_changes, m_impl->d_changes,
                                        sizeof(int), cudaMemcpyDeviceToHost,
                                        m_impl->compute_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->compute_stream));
        nchange = *(m_impl->h_changes);

        if (step % printfreq == 0) {
            printf("percolate: number of changes on step %d is %d\n",
                   step, nchange);
        }

        std::swap(d_next, d_current);
        step++;
    }

    // Record computation end and calculate time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->stop_compute, m_impl->compute_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->stop_compute));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, m_impl->start_compute, m_impl->stop_compute));
    m_impl->compute_time += milliseconds;

    // Ensure final state is in d_state
    if (d_current != m_impl->d_state) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->d_state, d_current,
                                        m_impl->size() * sizeof(int),
                                        cudaMemcpyDeviceToDevice,
                                        m_impl->compute_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->compute_stream));
    }

    // Print timing statistics
    m_impl->print_timing_stats();
}
