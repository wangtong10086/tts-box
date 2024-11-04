# CMake minimum version
cmake_minimum_required(VERSION 3.18)

# Project name and enable CUDA
project(MyCudaProject LANGUAGES CXX CUDA)

# Specify the CUDA standard version
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Add the executable file
add_executable(test test.cu)
