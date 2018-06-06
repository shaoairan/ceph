#ifndef GF_TYPES_H
#define GF_TYPES_H

__host__ __device__ class gf_w8_log_gpu {
public:
        unsigned  char*        g_log;
        unsigned  char*        g_anti_log;
        unsigned  char*        g_inv;
 };

 #endif