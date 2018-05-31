#ifndef GF_BASE_H
#define GF_BASE_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define GF_FIELD_WIDTH_8 		8
#define GF_FIELD_SIZE_8			256
#define GF_HALF_SIZE_8			16
#define GF_MULT_GROUP_SIZE_8	255

extern __shared__ char sh_log[GF_FIELD_SIZE_8];
extern __shared__ char sh_antilog[GF_FIELD_SIZE_8*2];
extern __shared__ char sh_inv[GF_FIELD_SIZE_8];

extern __constant__ char c_log[GF_FIELD_SIZE_8];
extern __constant__ char c_antilog[GF_FIELD_SIZE_8*2];
extern __constant__ char c_inv[GF_FIELD_SIZE_8];

__device__ __inline__ void load_tables(uint3 threadIdx, const dim3 blockDim);

__device__  inline char galois_single_divide_gpu_logtable_w8(char a, char b);

__device__ inline char galois_single_multiply_gpu_logtable_w8(char a, char b);

__device__ inline char galois_single_inverse_gpu_logtable_w8 (char a);

int copy_log_to_gpu_w8( void* log, void* anti_log, void* inv, cudaStream_t stream = 0 );

#endif