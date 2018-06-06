cmake_minimum_required(VERSION 3.9.6)

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)

if (UNIX)
    if(APPLE)
        message(FATAL_ERROR "System not supported, try linux!")
    else()
        message(STATUS "========================marky: adding nvcc flags...==========================")
        list(APPEND CMAKE_CXX_FLAGS " -O2 -mssse3 -fno-operator-names")
        list(APPEND CMAKE_CUDA_FLAGS " -lineinfo -arch=sm_50 --compiler-options -O2 -DVERBOSE")
    endif()
else()
    message(FATAL_ERROR "System not supported, try linux!")
endif()



add_library(gf_bridge OBJECT gf_bridge.c )


add_library(cephCudaLibTest STATIC clmsr_gpu.cu gf_base.cu jerasure_base.cu)
target_compile_features(cephCudaLibTest PUBLIC cxx_std_11)