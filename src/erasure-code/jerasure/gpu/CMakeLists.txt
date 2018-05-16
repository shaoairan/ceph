cmake_minimum_required(VERSION 3.5.1)
project(cephCudaLibTest C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (UNIX)
    if(APPLE)
        message(FATAL_ERROR "System not supported, try linux!")
    else()
        message(STATUS "marky: adding nvcc flags...")
        list(APPEND CMAKE_CXX_FLAGS " -O2 -mssse3 -fPIC ")
        list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
    endif()
else()
    message(FATAL_ERROR "System not supported, try linux!")
endif()


#set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(cephCudaLibTest SHARED mylibrary.cu mylibrary.h)
