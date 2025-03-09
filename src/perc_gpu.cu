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

// Configuration constants
constexpr int BLOCK_SIZE = 32;  // Increased from 16 to 32 for better occupancy
constexpr int printfreq = 100;
constexpr int SHARED_MEM_PADDING = 1;  // Padding to avoid bank conflicts

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

// Optimized CUDA kernel using shared memory and improved memory access patterns
__global__ void percolate_kernel(int M, int N, const int* __restrict__ state, 
                                int* __restrict__ next, int* changes) {
    extern __shared__ int shared_state[];
    
    // Calculate global and local indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * (blockDim.x - 2);  // Overlap blocks by 1 cell
    const int by = blockIdx.y * (blockDim.y - 2);
    const int i = by + ty;
    const int j = bx + tx;
    
    // Shared memory dimensions including halo
    const int smem_pitch = BLOCK_SIZE + SHARED_MEM_PADDING;
    const int smem_idx = ty * smem_pitch + tx;
    
    // Load data into shared memory including halo regions
    if (i <= M && j <= N) {
        const int global_idx = i * (N + 2) + j;
        shared_state[smem_idx] = state[global_idx];
        
        // Load halo regions if thread is on block boundary
        if (tx == 0 && j > 0) {
            shared_state[smem_idx - 1] = state[global_idx - 1];
        }
        if (tx == blockDim.x - 1 && j < N) {
            shared_state[smem_idx + 1] = state[global_idx + 1];
        }
        if (ty == 0 && i > 0) {
            shared_state[smem_idx - smem_pitch] = state[global_idx - (N + 2)];
        }
        if (ty == blockDim.y - 1 && i < M) {
            shared_state[smem_idx + smem_pitch] = state[global_idx + (N + 2)];
        }
    }
    
    __syncthreads();
    
    // Process only interior points
    if (i >= 1 && i <= M && j >= 1 && j <= N) {
        const int oldval = shared_state[smem_idx];
        int newval = oldval;
        
        if (oldval != 0) {
            // Use shared memory for neighbor access
            newval = max(newval, shared_state[smem_idx - smem_pitch]);  // Up
            newval = max(newval, shared_state[smem_idx + smem_pitch]);  // Down
            newval = max(newval, shared_state[smem_idx - 1]);          // Left
            newval = max(newval, shared_state[smem_idx + 1]);          // Right
            
            // Write result to global memory
            next[i * (N + 2) + j] = newval;
            
            // Use warp-level primitives to reduce atomic operations
            unsigned mask = __ballot_sync(0xffffffff, newval != oldval);
            if (mask) {
                if (threadIdx.x == 0) {
                    atomicAdd(changes, __popc(mask));
                }
            }
        } else {
            next[i * (N + 2) + j] = 0;
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
    
    // CUDA streams for async operations
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    
    // CUDA events for timing
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaEvent_t transfer_start;
    cudaEvent_t transfer_stop;
    
    float kernel_ms;    // Kernel execution time
    float transfer_ms;  // Transfer time

    Impl(int m, int n) : M(m), N(n), kernel_ms(0), transfer_ms(0) {
        // Allocate pinned memory for better transfer speeds
        CHECK_CUDA_ERROR(cudaHostAlloc(&state, size() * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&tmp, size() * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_changes, sizeof(int), cudaHostAllocDefault));
        
        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_state, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(int)));
        
        // Create CUDA streams
        CHECK_CUDA_ERROR(cudaStreamCreate(&compute_stream));
        CHECK_CUDA_ERROR(cudaStreamCreate(&transfer_stream));
        
        // Create timing events
        CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&transfer_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&transfer_stop));
    }
    
    ~Impl() {
        // Free pinned memory
        CHECK_CUDA_ERROR(cudaFreeHost(state));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp));
        CHECK_CUDA_ERROR(cudaFreeHost(h_changes));
        
        // Free device memory
        cudaFree(d_state);
        cudaFree(d_tmp);
        cudaFree(d_changes);
        
        // Destroy streams and events
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaEventDestroy(transfer_start);
        cudaEventDestroy(transfer_stop);
    }

    int size() const {
        return (M + 2) * (N + 2);
    }
};

GpuRunner::GpuRunner(int M, int N) : m_impl(std::make_unique<Impl>(M, N)) {
}

GpuRunner::~GpuRunner() = default;

void GpuRunner::copy_in(int const* source) {
    // Record transfer start time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->transfer_start, m_impl->transfer_stream));
    
    // Copy to pinned memory then to device asynchronously
    std::copy(source, source + m_impl->size(), m_impl->state);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->d_state, m_impl->state,
                                    m_impl->size() * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    m_impl->transfer_stream));
    
    // Record transfer end time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->transfer_stop, m_impl->transfer_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->transfer_stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&m_impl->transfer_ms,
                                         m_impl->transfer_start,
                                         m_impl->transfer_stop));
}

void GpuRunner::copy_out(int* dest) const {
    // Record transfer start time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->transfer_start, m_impl->transfer_stream));
    
    // Copy from device to pinned memory then to destination
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->state, m_impl->d_state,
                                    m_impl->size() * sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    m_impl->transfer_stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->transfer_stream));
    std::copy(m_impl->state, m_impl->state + m_impl->size(), dest);
    
    // Record transfer end time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->transfer_stop, m_impl->transfer_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->transfer_stop));
    float transfer_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&transfer_ms,
                                         m_impl->transfer_start,
                                         m_impl->transfer_stop));
    m_impl->transfer_ms += transfer_ms;
}

void GpuRunner::run() {
    int const M = m_impl->M;
    int const N = m_impl->N;
    
    // Calculate optimal grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 3) / (BLOCK_SIZE - 2),
                 (M + BLOCK_SIZE - 3) / (BLOCK_SIZE - 2));
    
    // Calculate shared memory size with padding
    size_t smem_size = (BLOCK_SIZE + SHARED_MEM_PADDING) * 
                       (BLOCK_SIZE + SHARED_MEM_PADDING) * sizeof(int);

    int const maxstep = 4 * std::max(M, N);
    int step = 1;
    int nchange = 1;

    // Use pointers to device buffers
    int* d_current = m_impl->d_state;
    int* d_next = m_impl->d_tmp;
    
    // Record kernel start time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->start_event, m_impl->compute_stream));

    while (nchange && step <= maxstep) {
        // Reset change counter
        CHECK_CUDA_ERROR(cudaMemsetAsync(m_impl->d_changes, 0, sizeof(int),
                                        m_impl->compute_stream));
        
        // Launch kernel with shared memory
        percolate_kernel<<<gridDim, blockDim, smem_size, m_impl->compute_stream>>>
            (M, N, d_current, d_next, m_impl->d_changes);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Get number of changes asynchronously
        CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->h_changes, m_impl->d_changes,
                                        sizeof(int), cudaMemcpyDeviceToHost,
                                        m_impl->compute_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_impl->compute_stream));
        nchange = *(m_impl->h_changes);

        if (step % printfreq == 0) {
            std::printf("percolate: number of changes on step %d is %d\n",
                       step, nchange);
        }

        std::swap(d_next, d_current);
        step++;
    }
    
    // Record kernel stop time
    CHECK_CUDA_ERROR(cudaEventRecord(m_impl->stop_event, m_impl->compute_stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(m_impl->stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&m_impl->kernel_ms,
                                         m_impl->start_event,
                                         m_impl->stop_event));

    // Ensure final state is in d_state
    if (d_current != m_impl->d_state) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(m_impl->d_state, d_current,
                                        m_impl->size() * sizeof(int),
                                        cudaMemcpyDeviceToDevice,
                                        m_impl->compute_stream));
    }
    
    // Print timing information
    printf("\nPerformance Summary:\n");
    printf("Kernel execution time: %.3f ms\n", m_impl->kernel_ms);
    printf("Memory transfer time: %.3f ms\n", m_impl->transfer_ms);
    printf("Total time: %.3f ms\n", m_impl->kernel_ms + m_impl->transfer_ms);
}
