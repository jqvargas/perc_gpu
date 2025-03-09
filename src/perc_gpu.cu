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

// Increased block size for better V100 GPU occupancy
// V100 has 80 SMs, each supporting up to 2048 threads
// 32x32 = 1024 threads per block is a good choice for V100
constexpr int BLOCK_SIZE = 32;  // Thread block size (32x32)
constexpr int printfreq = 100;

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) check_cuda( (val), #val, __FILE__, __LINE__ )
template<typename T>
void check_cuda(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n",
                file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel for percolation step
__global__ void percolate_kernel(int M, int N, const int* __restrict__ state, 
                                int* __restrict__ next, int* changes) {
    // Calculate indices with adjusted block size
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= M && j <= N) {
        int idx = i * (N + 2) + j;
        int oldval = state[idx];
        int newval = oldval;

        if (oldval != 0) {
            // Check neighbors and update maximum
            newval = max(newval, state[idx - (N + 2)]);  // Up
            newval = max(newval, state[idx + (N + 2)]);  // Down
            newval = max(newval, state[idx - 1]);        // Left
            newval = max(newval, state[idx + 1]);        // Right

            next[idx] = newval;
            if (newval != oldval) {
                atomicAdd(changes, 1);
            }
        } else {
            next[idx] = 0;
        }
    }
}

struct GpuRunner::Impl {
    int M;
    int N;
    int* state;      // Host memory (pinned)
    int* tmp;        // Host memory (pinned)
    int* d_state;    // Device memory
    int* d_tmp;      // Device memory
    int* d_changes;  // Device memory for counting changes
    int* h_changes;  // Host memory for changes (pinned)

    // Timing events
    cudaEvent_t start_compute;
    cudaEvent_t stop_compute;
    cudaEvent_t start_h2d;
    cudaEvent_t stop_h2d;
    cudaEvent_t start_d2h;
    cudaEvent_t stop_d2h;

    // Timing results (in milliseconds)
    float compute_time;
    float h2d_time;
    float d2h_time;

    // CUDA streams
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
