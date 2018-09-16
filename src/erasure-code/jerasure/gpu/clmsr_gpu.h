#ifndef CUDALIBTEST_LIBRARY_H
#define CUDALIBTEST_LIBRARY_H

#include<map>
#include<set>
#include<vector>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "gf_types.h"
#include "cl_bridge.h"

using namespace std;

#define STREAM_NUM 3
#define EVENT_NUM  3


extern "C" {
int docal();
}

typedef enum MdsType{
  RS=0,
  CRS=1
}mdsType_t;


struct ClmsrProfile
{
public:
	unsigned chunkSize;
	unsigned subChunkSize;
	int sub_chunk_no;

	int gamma;
	int *matrix;

	int q;
	int t;
	int d;

	int k;
	int m;
	int w;

	int nu;

	MdsType mdsType;

	ClmsrProfile( int q_,int t_,int d_, int sub_chunk_no_,\
		int k_,int m_,int w_,int nu_,\
		int gamma_,int* matrix_, \
		unsigned chunkSize_, unsigned subChunkSize_, MdsType mdsType_ );

	
};

struct DeviceInfo
{
public:
	int deviceCount;
	cudaDeviceProp *device;

	DeviceInfo();
	~DeviceInfo();
};

class ClmsrGpu{


/*map<int,char*> &repaired_data, 
set<int> &aloof_nodes,
map<int, char*> &helper_data,
int repair_blocksize, 
map<int,int> &repair_sub_chunks_ind
*/
public:

	char** B_buf;

	static bool statusMark;
	ClmsrProfile clmsrProfile;
	DeviceInfo deviceInfo;

	ClmsrGpu( ClmsrProfile clmsrProfile_ );
	~ClmsrGpu();
	int pinAllMemoryForRepair(  map<int,char*>& repaired_data, int sizeRepair,  map<int,char*>& helper_data,  int sizeHelper );
	int unpinAllMemoryForRepair(  map<int,char*>& repaired_data,   map<int,char*>& helper_data );
	int pinAllMemoryForDecode(  char** data_ptrs, int sizeData,  char** code_ptrs,  int sizeCode );
	int unpinAllMemoryForDecode(  char** data_ptrs,   char** code_ptrs );

private:
	inline void pinMemory( map<int,char*> map, int size );
	inline void unpinMemory( map<int,char*> map );

};

class SingleGpuRoute
{
public:

	int *matrix_gpu;
	ClmsrGpu* clmsrGpuP;
	ClmsrProfile* clmsrProfileP;
	int deviceId;
	cudaDeviceProp &deviceProp;
	//complie debug
	cudaStream_t *streams;//0: in, 1: cal, 3 out
	cudaEvent_t *events;
	int subSubChunkStart;
	int subSubChunkSize;

	int pieceKernelGridSize;
	int pieceKernelBlockSize;

	int planeKernelGridSize;
	int planeKernelBlockSize;

	gf_w8_log_gpu gf_table;
	
private:
	int pieceCount;
	int __getPieceSize( int i );


public:
	SingleGpuRoute( int deviceId_, ClmsrGpu* ClmsrGpuP_, int subSubChunkStart_, int subSubChunkSize_ );
	~SingleGpuRoute();
	
	void init();
	int doRepair(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                        map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind, char** B_buf );

	/*int doDecode( int* erasure_locations, char** data_ptrs, char** code_ptrs, int* erased, \
                            int num_erasures, int* order, int* weight_vec, int max_weight, int size, char ** B_buf);*/

	int doDecode( int* erasure_locations, char** data_ptrs, char** code_ptrs, int* erased, \
                            int num_erasures, int* order, int* weight_vec, int max_weight, int size );
    void deinit();
	int init_gf_log_w8_gpu( cudaStream_t stream = 0 );

	void compareGf();
};

#endif