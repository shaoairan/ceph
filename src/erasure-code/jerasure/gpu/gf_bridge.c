#include "gf_bridge.h"
#include "jerasure.h"
#include "galois.h"
#include "gf_w8.h"
#include <stdio.h>
#include <stdlib.h>
#include "gf_complete.h"

#define MAX_GF_ARRAY_SIZE 64

gf_t * gfp_w8_log = NULL;



gfp_w8_log_gpu get_w8_log_tables()
{
    gfp_w8_log_gpu table_p;
    if( gfp_w8_log == NULL )
    {
/*      gf_t* galois_init_field(int w,
                        int mult_type,
                        int region_type,
                        int divide_type,
                        uint64_t prim_poly,
                        int arg1,
                        int arg2)*/

      //printf(" get gfp failed, creating gf-field!! ");
      gfp_w8_log = galois_init_field(8,
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
  ltd = (struct gf_w8_logtable_data *) ((gf_internal_t *) gfp_w8_log->scratch)->private;

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