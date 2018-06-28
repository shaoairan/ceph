#include "gf_base.h"
#include <iostream>

extern "C" {
#include "gf_bridge.h"
}

int copy_log_to_gpu_w8( gf_w8_log_gpu& gf_table,  cudaStream_t stream )
{
  	gfp_w8_log_gpu table_p = get_w8_log_tables();
 
	//print_table(0, log, anti_log, inv);
  	//printf("geted, get_w8_log_tables\n");



	//debug
	/*cudaMemcpyToSymbolAsync(c_log, 		table_p.log,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_antilog, 	table_p.anti_log,	GF_FIELD_SIZE_8_GPU*2, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_inv, 		table_p.inv,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);*/

	cudaMemcpy(gf_table.g_log, 		table_p.log,		GF_FIELD_SIZE_8_GPU, cudaMemcpyHostToDevice);
	cudaMemcpy(gf_table.g_anti_log, table_p.anti_log,	GF_FIELD_SIZE_8_GPU*2, cudaMemcpyHostToDevice);
	cudaMemcpy(gf_table.g_inv, 		table_p.inv,		GF_FIELD_SIZE_8_GPU, cudaMemcpyHostToDevice);
	

	//std::cout << " your pt:  " << (int)(&c_log[0]) << std::endl;
	//printf("your pt1:   %d\n", &c_log[0] );

/*	for( int i = 0; i < 256; i ++ )
	{
		printf("origin---i: %d:\t%d\t %d\n", i, table_p.log[i], table_p.anti_log[i] );
	}

	for( int i = 256; i < 256*2; i ++ )
	{
		printf("origin---i: %d:\t%d\n", i, table_p.anti_log[i] );
	}*/

	return 0;
}


