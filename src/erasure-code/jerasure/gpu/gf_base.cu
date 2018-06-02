#include "gf_base.h"

extern "C" {

#define private privateMe
#include "jerasure.h"
#include "galois.h"
#include "gf_w8.h"

#define private

}


int copy_log_to_gpu_w8( /* void* log, void* anti_log, void* inv,*/ cudaStream_t stream )
{
    gf_t * gfp = galois_get_field_ptr(8);
    if( gfp == NULL )
    {
/*      gf_t* galois_init_field(int w,
                        int mult_type,
                        int region_type,
                        int divide_type,
                        uint64_t prim_poly,
                        int arg1,
                        int arg2)*/

      printf(" get gfp failed, creating gf-field inited!! ");
      gfp = galois_init_field(8,
                        GF_MULT_LOG_TABLE,
                        GF_REGION_DEFAULT,
                        GF_DIVIDE_DEFAULT,
                        0,
                        0,
                        0);
      //return 1;
    }

/*    struct gf_w8_logtable_data {
        uint8_t         log_tbl[GF_FIELD_SIZE];
        uint8_t         antilog_tbl[GF_FIELD_SIZE * 2];
        uint8_t         inv_tbl[GF_FIELD_SIZE];
    };*/

  struct gf_w8_logtable_data *ltd;
  ltd = (struct gf_w8_logtable_data *) ((gf_internal_t *) gfp->scratch)->privateMe;
  char* log = (char*)ltd->log_tbl;
  char* anti_log = (char*)ltd->antilog_tbl;
  char* inv = (char*)ltd->inv_tbl;
 
	cudaMemcpyToSymbolAsync(c_log, 		log,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_antilog, 	anti_log,	GF_FIELD_SIZE_8_GPU*2, 	0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(c_inv, 		inv,		GF_FIELD_SIZE_8_GPU, 	0, cudaMemcpyHostToDevice, stream);

	return 0;
}
