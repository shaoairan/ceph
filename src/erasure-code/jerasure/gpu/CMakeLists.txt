cmake_minimum_required(VERSION 3.9.6)

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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