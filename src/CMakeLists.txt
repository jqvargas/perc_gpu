# Copyright (C) 2025, Rupert Nash, The University of Edinburgh.

# All rights reserved.

# This file is provided to you to complete an assessment and for
# subsequent private study. It may not be shared and, in particular,
# may not be posted on the internet. Sharing this or any modified
# version may constitute academic misconduct under the University's
# regulations.

cmake_minimum_required(VERSION 3.18)
project(perc CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

find_package(PNG REQUIRED)

add_executable(test util.cpp perc_cpu.cpp perc_gpu.cu  test.cpp)

target_link_libraries(test PRIVATE PNG::PNG)

set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS test)
