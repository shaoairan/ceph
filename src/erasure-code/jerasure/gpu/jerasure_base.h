#ifndef JERASURE_BASE_H
#define JERASURE_BASE_H

int erasures_to_erased_gpu(int k, int m, int *erasures, int *erased );

int full_erased_list_data( int k, int m, int * erasure_loc_data, int * erased );

int full_erased_list_coding( int k, int m, int * erasure_loc_coding, int * erased );

#endif