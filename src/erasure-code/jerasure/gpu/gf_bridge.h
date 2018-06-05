#ifndef GF_BRIDGE_H
#define GF_BRIDGE_H

typedef struct{
        unsigned  char*        log;
        unsigned  char*        anti_log;
        unsigned  char*        inv;
 }gfp_w8_log_gpu;

gfp_w8_log_gpu get_w8_log_tables();

#endif