cmake_minimum_required(VERSION 3.28)

# without following set command, /usr/bin/nvcc is executed
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(rawML LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(tests)
