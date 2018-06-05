#include "gf_base.h"
#include <iostream>

extern "C" {
#include "gf_bridge.h"
}

__constant__ unsigned char c_log[GF_FIELD_SIZE_8_GPU];
__constant__ unsigned char c_antilog[GF_FIELD_SIZE_8_GPU*2];
__constant__ unsigned char c_inv[GF_FIELD_SIZE_8_GPU];

int copy_log_to_gpu_w8( /* void* log, void* anti_log, void* inv,*/ cudaStream_t stream )
{
  	gfp_w8_log_gpu table_p = get_w8_log_tables();
 
	//print_table(0, log, anti_log, inv);
  	printf("geted, get_w8_log_tables\n");



	//debug
	/*cudaMemcpyToSymbolAsync(c_log, 		table_p.log,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_antilog, 	table_p.anti_log,	GF_FIELD_SIZE_8_GPU*2, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_inv, 		table_p.inv,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);*/

	cudaMemcpyToSymbol(c_log, 		table_p.log,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_antilog, 	table_p.anti_log,	GF_FIELD_SIZE_8_GPU*2, 	0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_inv, 		table_p.inv,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice);
	

	//std::cout << " your pt:  " << (int)(&c_log[0]) << std::endl;
	printf("your pt1:   %d\n", &c_log[0] );

/*	for( int i = 0; i < 256; i ++ )
	{
		printf("i: %d:\t%d\t %d\n", i, table_p.log[i], table_p.anti_log[i] );
	}

	for( int i = 256; i < 256*2; i ++ )
	{
		printf("i: %d:\t%d\n", i, table_p.anti_log[i] );
	}
*/
	return 0;
}
