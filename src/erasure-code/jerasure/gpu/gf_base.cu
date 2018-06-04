#include "gf_base.h"

extern "C" {
#include "gf_bridge.h"
}


int copy_log_to_gpu_w8( /* void* log, void* anti_log, void* inv,*/ cudaStream_t stream )
{
  char* log;
  char* anti_log;
  char* inv;

  get_w8_log_tables( log, anti_log, inv );
 
	cudaMemcpyToSymbolAsync(c_log, 		log,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_antilog, 	anti_log,	GF_FIELD_SIZE_8_GPU*2, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_inv, 		inv,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);

	return 0;
}
