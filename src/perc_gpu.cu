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

constexpr int printfreq = 100;

// Do the 2D indexing into the array.
//
// Assumes that you have a variable `N` in scope specifying the the
// size of the non-halo part of the grid.
#define get(array, i, j) array[(i)*(N+2) + j]

// Perform a single step of the algorithm.
//
// For each point (if fluid), set it to the maximum of itself and the
// four von Neumann neighbours.
//
// Returns the total number of changed cells.
static int percolate_gpu_step(int M, int N, int const* state, int* next) {
  int nchange = 0;

  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const oldval = get(state, i, j);
      int newval = oldval;

      // 0 => solid, so do nothing
      if (oldval != 0) {
	// Set next[i][j] to be the maximum value of state[i][j] and
	// its four nearest neighbours
	newval = std::max(newval, get(state, i-1, j  ));
	newval = std::max(newval, get(state, i+1, j  ));
	newval = std::max(newval, get(state, i  , j-1));
	newval = std::max(newval, get(state, i  , j+1));

	if (newval != oldval) {
	  ++nchange;
	}
      }

      next[(i)*(N+2) + j] = newval;
    }
  }
  return nchange;
}


struct GpuRunner::Impl {
  // Here you can store any parameters or data needed for
  // implementation of your version.

  int M;
  int N;
  int* state;
  int* tmp;

  Impl(int m, int n) : M(m), N(n) {
    state = new int[size()];
    tmp = new int[size()];
  }
    
  ~Impl() {
    delete[] state;
    delete[] tmp;
  }

  int size() const {
    return (M + 2)*(N + 2);
  }
};

GpuRunner::GpuRunner(int M, int N) : m_impl(std::make_unique<Impl>(M, N)) {
}
GpuRunner::~GpuRunner() = default;

void GpuRunner::copy_in(int const* source) {
  std::copy(source, source + m_impl->size(), m_impl->state);
}

void GpuRunner::copy_out(int* dest) const {
  std::copy(m_impl->state, m_impl->state + m_impl->size(), dest);
}

// Given an array, state, of size (M+2) x (N+2) with a halo of zeros,
// iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
void GpuRunner::run() {
  int const npoints = m_impl->size();
  int const M = m_impl->M;
  int const N = m_impl->N;
  // Copy the initial state to the temp, only the halos are
  // *required*, but much easier this way!
  std::memcpy(m_impl->tmp, m_impl->state, sizeof(int) * m_impl->size());

  int const maxstep = 4 * std::max(M, N);
  int step = 1;
  int nchange = 1;

  // Use pointers to the buffers (which we swap below) to avoid copies.
  int* current = m_impl->state;
  int* next = m_impl->tmp;

  while (nchange && step <= maxstep) {
    nchange = percolate_gpu_step(M, N, current, next);

    //  Report progress every now and then
    if (step % printfreq == 0) {
      std::printf("percolate: number of changes on step %d is %d\n",
		  step, nchange);
    }

    // Swap the pointers for the next iteration
    std::swap(next, current);
    step++;
  }

  // Answer now in `current`, make sure this one is in `state`
  m_impl->state = current;
  m_impl->tmp = next;
}
