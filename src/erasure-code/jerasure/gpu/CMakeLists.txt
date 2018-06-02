cmake_minimum_required(VERSION 3.9.6)
project(cephCudaLibTest C CUDA)

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

#set(CUDA_SEPARABLE_COMPILATION ON)

add_library(cephCudaLibTest STATIC clmsr_gpu.cu clmsr_gpu.h gf_base.cu jerasure_base.cu )

target_link_libraries(cephCudaLibTest Jerasure gf_complete)

target_compile_features(cephCudaLibTest PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties(cephCudaLibTest PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)