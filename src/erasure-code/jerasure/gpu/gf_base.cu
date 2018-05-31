#include "gf_base.h"


__shared__ char sh_log[GF_FIELD_SIZE_8];
__shared__ char sh_antilog[GF_FIELD_SIZE_8*2];
__shared__ char sh_inv[GF_FIELD_SIZE_8];

__constant__ char c_log[GF_FIELD_SIZE_8];
__constant__ char c_antilog[GF_FIELD_SIZE_8*2];
__constant__ char c_inv[GF_FIELD_SIZE_8];

__device__ __inline__ void load_tables(uint3 threadIdx, const dim3 blockDim) {
  /* Fully arbitrary routine for any blocksize and fetch size to load
   * the log and ilog tables into shared memory.
   */

   for( int i = threadIdx.x; i < 256; i += blockDim.x )
   {
   		sh_log[i] = c_log[i];
   		sh_ilog[i] = c_ilog[i];
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

int copy_log_to_gpu_w8( void* log, void* anti_log, void* inv, cudaStream_t stream = 0 )
{
	cudaMemcpyToSymbolAsync(c_log, 		log,		GF_FIELD_SIZE_8, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_antilog, 	anti_log,	GF_FIELD_SIZE_8*2, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_inv, 		inv,		GF_FIELD_SIZE_8, 	0, cudaMemcpyHostToDevice, stream);

	return 0;
}

__device__  inline char galois_single_divide_gpu_logtable_w8(char a, char b)
{
	return (a == 0 || b == 0) ? 0 : sh_ilog[sh_log[a] - sh_log[b] + (GF_MULT_GROUP_SIZE_8)];
}

__device__ inline char galois_single_multiply_gpu_logtable_w8(char a, char b)
{
	//todo: %256
	return (a == 0 || b == 0) ? 0 : sh_ilog[(unsigned)(sh_log[a] + sh_log[b])];
}

__device__ inline char galois_single_inverse_gpu_logtable_w8 (char a)
{
  return (ltd->sh_inv[a]);
}