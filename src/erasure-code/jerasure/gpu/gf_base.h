#ifndef GF_BASE_H
#define GF_BASE_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define GF_FIELD_WIDTH_8_GPU    8
#define GF_FIELD_SIZE_8_GPU     256
#define GF_HALF_SIZE_8_GPU      16
#define GF_MULT_GROUP_SIZE_8_GPU  255

#define get_arrs() 	static __shared__ unsigned char sh_log[GF_FIELD_SIZE_8_GPU]; static __shared__ unsigned char sh_antilog[GF_FIELD_SIZE_8_GPU*2]; static __shared__ unsigned char sh_inv[GF_FIELD_SIZE_8_GPU]
#define init_table( threadIdx, blockDim )	load_tables(threadIdx, blockDim, sh_log, sh_antilog, sh_inv)

#define galois_single_divide_gpu_logtable_w8( a, b )		(a == 0 || b == 0) ? 0 : sh_antilog[sh_log[a] - sh_log[b] + (GF_MULT_GROUP_SIZE_8_GPU)]
#define galois_single_multiply_gpu_logtable_w8( a, b )		(a == 0 || b == 0) ? 0 : sh_antilog[(unsigned)(sh_log[a] + sh_log[b])]
#define galois_single_inverse_gpu_logtable_w8 (a)			sh_inv[a]
#define set_gf_table() get_arrs(); init_table( threadIdx, blockDim )


extern __constant__ unsigned char c_log[GF_FIELD_SIZE_8_GPU];
extern __constant__ unsigned char c_antilog[GF_FIELD_SIZE_8_GPU*2];
extern __constant__ unsigned char c_inv[GF_FIELD_SIZE_8_GPU];

/*__device__ inline void load_tables(uint3 threadIdx, const dim3 blockDim, char* sh_log, char* sh_antilog, char* sh_inv );

__device__  inline char galois_single_divide_gpu_logtable_w8(char a, char b);

__device__ inline char galois_single_multiply_gpu_logtable_w8(char a, char b);

__device__ inline char galois_single_inverse_gpu_logtable_w8 (char a);*/


/*__device__  inline char galois_single_divide_gpu_logtable_w8(char a, char b)
{
	return (a == 0 || b == 0) ? 0 : sh_antilog[sh_log[a] - sh_log[b] + (GF_MULT_GROUP_SIZE_8_GPU)];
}

__device__ inline char galois_single_multiply_gpu_logtable_w8(char a, char b)
{
	//todo: %256
	return (a == 0 || b == 0) ? 0 : sh_ilog[(unsigned)(sh_log[a] + sh_log[b])];
}

__device__ inline char galois_single_inverse_gpu_logtable_w8 (char a)
{
  return sh_inv[a];
}*/

__device__ inline void load_tables(uint3 threadIdx, const dim3 blockDim, unsigned char* sh_log, unsigned char*  sh_antilog, unsigned char* sh_inv ) {
  /* Fully arbitrary routine for any blocksize and fetch size to load
   * the log and ilog tables into shared memory.
   */

   	for( int i = threadIdx.x; i < 256; i += blockDim.x )
   	{
      sh_log[i] = c_log[i];
      sh_antilog[i] = c_antilog[i];
      sh_inv[i] = c_inv[i];
   	}

/*  int iters = ROUNDUPDIV(256,fetchsize);
  for (int i = 0; i < iters; i++) {
    if (i*fetchsize/SOF+threadIdx.x < 256/SOF) {
      int fetchit = threadIdx.x + i*fetchsize/SOF;
      ((fetch *)sh_log)[fetchit] = *(fetch *)(&gf_log_d[fetchit*SOF]);
      ((fetch *)sh_ilog)[fetchit] = *(fetch *)(&gf_ilog_d[fetchit*SOF]);
    }
  }*/
  //todo: syncthreads();
    __syncthreads();
    
}

__device__ inline void print_table( int idx, unsigned char* sh_log, unsigned char*  sh_antilog, unsigned char* sh_inv )
{
  if( idx == 0 )
  {
      for( int i = 0; i < 256; i ++ )
        {
            printf( "&====: %d: %d----%d\n", i, sh_log[i], sh_antilog[i]);
        }

        for( int i = 256; i < 256*2; i ++ )
        {
            printf( "&==&==: %d: %d\n", i, sh_antilog[i] );            
        }
  }
}

int copy_log_to_gpu_w8( cudaStream_t stream = 0 );

#endif