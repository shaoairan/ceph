#include "gf_bridgeout.h"
#include <stdio.h>
#include <stdlib.h>

#include "jerasure.h"
#include "galois.h"
#include "gf_w8.h"
#include "gf_complete.h"


#define MAX_GF_ARRAY_SIZE 64
#define GF_FIELD_WIDTH_8    8
#define GF_FIELD_SIZE_8     256
#define GF_HALF_SIZE_8      16
#define GF_MULT_GROUP_SIZE_8  255


gf_t * gfp_w8_logOut = NULL;

gfp_w8_log_gpuOut table_p_global;

inline unsigned char galois_single_divide_logtable_w8_bridgeOut( unsigned char a, unsigned char b ){
  return (((a) == 0 || (b) == 0) ? 0 : table_p_global.anti_log[table_p_global.log[(a)] - table_p_global.log[(b)] + (GF_MULT_GROUP_SIZE_8)]);
}

inline unsigned char galois_single_multiply_logtable_w8_bridgeOut( unsigned char a, unsigned char b ){
  return (((a) == 0 || (b) == 0) ? 0 : table_p_global.anti_log[(unsigned)(table_p_global.log[(a)] + table_p_global.log[(b)])]);
}

inline unsigned char galois_single_inverse_gpu_logtable_w8_bridgeOut( unsigned char a ) {
  return (table_p_global.inv[(a)]);
}

void galois_w08_region_multiply_bridgeOut(  char * sptr,  int num, int size,  char* dptr, int add){
  if(add) {
    for( int i = 0; i < size; i ++ )
    {
      dptr[i] = galois_single_multiply_logtable_w8_bridgeOut( sptr[i], num ) ^ dptr[i];
    }
  } else {
    for( int i = 0; i < size; i ++ )
    {
      dptr[i] = galois_single_multiply_logtable_w8_bridgeOut( sptr[i], num );
    }
  }
}

void galios_matrix_dotprod_bridgeOut(int k, int w, int *matrix_row,
                          int *src_ids, int dest_id,
                          char **data_ptrs, char **coding_ptrs, int size) {
    int init = 0;
    char *dptr, *sptr;
    dptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];

    for (int i = 0; i < k; i++) {
      if (src_ids == NULL) {
        sptr = data_ptrs[i];
      } else if (src_ids[i] < k) {
        sptr = data_ptrs[src_ids[i]];
      } else {
        sptr = coding_ptrs[src_ids[i]-k];
      }

      galois_w08_region_multiply_bridgeOut(sptr, matrix_row[i], size, dptr, init);

      init = 1;
    }
}



gfp_w8_log_gpuOut get_w8_log_tables_bridgeOut()
{
    gfp_w8_log_gpuOut table_p;
    if( gfp_w8_logOut == NULL )
    {
/*      gf_t* galois_init_field(int w,
                        int mult_type,
                        int region_type,
                        int divide_type,
                        uint64_t prim_poly,
                        int arg1,
                        int arg2)*/

      //printf(" get gfp failed, creating gf-field!! ");
      gfp_w8_logOut = galois_init_field(8,
                        GF_MULT_LOG_TABLE,
                        GF_REGION_DEFAULT,
                        GF_DIVIDE_DEFAULT,
                        0,
                        0,
                        0);
      //return 1;
    }

    //printf(" get gfp succeed~~~~~~~~~~~~~~~ ");

/*    struct gf_w8_logtable_data {
        uint8_t         log_tbl[GF_FIELD_SIZE];
        uint8_t         antilog_tbl[GF_FIELD_SIZE * 2];
        uint8_t         inv_tbl[GF_FIELD_SIZE];
    };*/

  struct gf_w8_logtable_data *ltd;
  ltd = (struct gf_w8_logtable_data *) ((gf_internal_t *) gfp_w8_logOut->scratch)->private;

/*  for( int i = 0; i < 256; i ++ )
  {
    printf("i: %d:\t%d\t %d\n", i, ltd->log_tbl[i], ltd->antilog_tbl[i] );
  }

  for( int i = 256; i < 256*2; i ++ )
  {
    printf("i: %d:\t%d\n", i, ltd->antilog_tbl[i] );
  }*/

  table_p.log = (unsigned char*)ltd->log_tbl;
  table_p.anti_log = (unsigned char*)ltd->antilog_tbl;
  table_p.inv = (unsigned char*)ltd->inv_tbl;

  return table_p;
}