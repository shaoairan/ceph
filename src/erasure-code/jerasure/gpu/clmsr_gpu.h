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

	map<int,char*> &repaired_data; 
	set<int> &aloof_nodes;
	map<int, char*> &helper_data;
	int repair_blocksize;
	map<int,int> &repair_sub_chunks_ind;
	char** B_buf;
	int *matrix_gpu;

	static bool statusMark;
	ClmsrProfile clmsrProfile;
	DeviceInfo deviceInfo;

	ClmsrGpu( map<int,char*> &repaired_data_, set<int> &aloof_nodes_, map<int, char*> &helper_data_, \
		int repair_blocksize_, map<int,int> &repair_sub_chunks_ind_, ClmsrProfile clmsrProfile_ );
	~ClmsrGpu();


private:
	inline void pinMemory( map<int,char*> map, int size );
	inline void unpinMemory( map<int,char*> map );

};

class SingleGpuRoute
{
public:
	ClmsrGpu* clmsrGpuP;
	ClmsrProfile* clmsrProfileP;
	int deviceId;
	cudaDeviceProp &deviceProp;
	//complie debug
	cudaStream_t streams[STREAM_NUM];//0: in, 1: cal, 3 out
	cudaEvent_t events[EVENT_NUM];
	int subSubChunkStart;
	int subSubChunkSize;

	int pieceKernelGridSize;
	int pieceKernelBlockSize;

	int planeKernelGridSize;
	int planeKernelBlockSize;
	
private:
	int pieceCount;
	int __getPieceSize( int i );


public:
	SingleGpuRoute( int deviceId_, ClmsrGpu* ClmsrGpuP_, int subSubChunkStart_, int subSubChunkSize_ );
	~SingleGpuRoute();
	
	void init();
	int doRepair(map<int,char*> &repaired_data, set<int> &aloof_nodes,
                           map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind, char** B_buf );
	int doDecode();
	void deinit();
};

#endif