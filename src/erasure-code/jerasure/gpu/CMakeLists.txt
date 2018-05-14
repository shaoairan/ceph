cmake_minimum_required(VERSION 3.5.1)
project(cephCudaLibTest C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# step 2. set compile flag: nppi, -arch
if (WIN32)
    set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS -fPIC ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
endif()

if (UNIX)
    if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -fPIC ")
        set(APPEND CUDA_NVCC_FLAGS "-lnppi -lnpps -lnppc;-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mssse3 -fPIC ")
        set(CUDA_NVCC_FLAGS "-lnppi -lnppc -lnpps;-lineinfo;-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
    endif()
endif()

include_directories(
        /usr/local/cuda/include
)

link_directories(
        /usr/local/cuda/lib64
)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(cephCudaLibTest SHARED mylibrary.cu mylibrary.h)