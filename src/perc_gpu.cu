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

constexpr int BLOCK_SIZE = 16;  // Thread block size (16x16)
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
__global__ void percolate_kernel(int M, int N, const int* state, int* next, int* changes) {
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
    int* state;      // Host memory
    int* tmp;        // Host memory
    int* d_state;    // Device memory
    int* d_tmp;      // Device memory
    int* d_changes;  // Device memory for counting changes
    int* h_changes;  // Host memory for changes

    Impl(int m, int n) : M(m), N(n) {
        state = new int[size()];
        tmp = new int[size()];
        h_changes = new int;

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_state, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_changes, sizeof(int)));
    }
    
    ~Impl() {
        delete[] state;
        delete[] tmp;
        delete h_changes;
        
        // Free device memory
        cudaFree(d_state);
        cudaFree(d_tmp);
        cudaFree(d_changes);
    }

    int size() const {
        return (M + 2) * (N + 2);
    }
};

GpuRunner::GpuRunner(int M, int N) : m_impl(std::make_unique<Impl>(M, N)) {
}

GpuRunner::~GpuRunner() = default;

void GpuRunner::copy_in(int const* source) {
    std::copy(source, source + m_impl->size(), m_impl->state);
    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(m_impl->d_state, m_impl->state, 
                               m_impl->size() * sizeof(int), 
                               cudaMemcpyHostToDevice));
}

void GpuRunner::copy_out(int* dest) const {
    // Copy result back from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(m_impl->state, m_impl->d_state,
                               m_impl->size() * sizeof(int),
                               cudaMemcpyDeviceToHost));
    std::copy(m_impl->state, m_impl->state + m_impl->size(), dest);
}

void GpuRunner::run() {
    int const M = m_impl->M;
    int const N = m_impl->N;
    
    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int const maxstep = 4 * std::max(M, N);
    int step = 1;
    int nchange = 1;

    // Use pointers to device buffers
    int* d_current = m_impl->d_state;
    int* d_next = m_impl->d_tmp;

    while (nchange && step <= maxstep) {
        // Reset change counter
        CHECK_CUDA_ERROR(cudaMemset(m_impl->d_changes, 0, sizeof(int)));
        
        // Launch kernel
        percolate_kernel<<<gridDim, blockDim>>>(M, N, d_current, d_next, m_impl->d_changes);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Get number of changes
        CHECK_CUDA_ERROR(cudaMemcpy(m_impl->h_changes, m_impl->d_changes,
                                   sizeof(int), cudaMemcpyDeviceToHost));
        nchange = *(m_impl->h_changes);

        // Report progress
        if (step % printfreq == 0) {
            std::printf("percolate: number of changes on step %d is %d\n",
                       step, nchange);
        }

        // Swap buffers
        std::swap(d_next, d_current);
        step++;
    }

    // Ensure final state is in d_state
    if (d_current != m_impl->d_state) {
        CHECK_CUDA_ERROR(cudaMemcpy(m_impl->d_state, d_current,
                                   m_impl->size() * sizeof(int),
                                   cudaMemcpyDeviceToDevice));
    }
}
