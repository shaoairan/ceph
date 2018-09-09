#include "gf_base.h"
#include <iostream>

extern "C" {
#include "gf_bridge.h"
}

int copy_log_to_gpu_w8( gf_w8_log_gpu& gf_table,  cudaStream_t stream )
{
  	gfp_w8_log_gpu table_p = get_w8_log_tables();
 
	cudaMemcpy(gf_table.g_log, 		table_p.log,		GF_FIELD_SIZE_8_GPU, cudaMemcpyHostToDevice);
	cudaMemcpy(gf_table.g_anti_log, table_p.anti_log,	GF_FIELD_SIZE_8_GPU*2, cudaMemcpyHostToDevice);
	cudaMemcpy(gf_table.g_inv, 		table_p.inv,		GF_FIELD_SIZE_8_GPU, cudaMemcpyHostToDevice);
	
	return 0;
}


