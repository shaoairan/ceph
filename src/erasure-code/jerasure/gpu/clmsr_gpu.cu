#include "clmsr_gpu.h"

#include <iostream>
#include <numeric>
#include <stdlib.h>
/*
 ============================================================================
 Name        : cudaTest.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

using namespace std;

#define FT(A) FunctionTest4 printFunctionName(#A)

class FunctionTest4
{
  static int tabs;
  std::string a;
  public:
    FunctionTest4( std::string a_ ):a(a_)
    {
      
      for( int i = 0; i < tabs; i ++ )
      {
          printf("\t");
      }
      std::cout << "entering:: " << a << "\n";
      tabs ++;
    }

    ~FunctionTest4()
    {
      tabs --;
      for( int i = 0; i < tabs; i ++ )
      {
          printf("\t");
      }
      std::cout << "leave:: " << a << "\n";
    }
};

int FunctionTest4::tabs = 4;


static bool CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

bool ClmsrGpu::statusMark = true;


/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < vectorSize)
        data[idx] = 1.0/data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
    float *rc = new float[size];
    float *gpuData;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

    static const int BLOCK_SIZE = 256;
    const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
    reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

    CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(gpuData));
    return rc;
}

float *cpuReciprocal(float *data, unsigned size)
{
    float *rc = new float[size];
    for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
    return rc;
}


void initialize(float *data, unsigned size)
{
    for (unsigned i = 0; i < size; ++i)
        data[i] = 1.5*(i+1);
}

int docal()
{
    std::cout << "Glad to see I'm here in cuda.so->clmsr_gpu.cu \n";
    static const int WORK_SIZE = 65530;
    float *data = new float[WORK_SIZE];

    initialize (data, WORK_SIZE);

    float *recCpu = cpuReciprocal(data, WORK_SIZE);
    float *recGpu = gpuReciprocal(data, WORK_SIZE);
    float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
    float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

    /* Verify the results */
    std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

    /* Free memory */
    delete[] data;
    delete[] recCpu;
    delete[] recGpu;

    return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static bool CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return true;
    else
    {
        std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
        ClmsrGpu::statusMark = false;
        exit(1);
    }
}

ClmsrProfile::ClmsrProfile( int q_,int t_,int d_,int sub_chunk_no_,\
        int k_,int m_,int w_,int nu_,\
        int gamma_,int* matrix_, \
        unsigned chunkSize_, unsigned subChunkSize_, MdsType mdsType_ ):\
        q(q_),t(t_),d(d_),\
        k(k_),m(m_),w(w_),nu(nu_),\
        gamma(gamma_),matrix(matrix_),\
        chunkSize(chunkSize_), subChunkSize(subChunkSize_), mdsType(mdsType_),sub_chunk_no(sub_chunk_no_)
        {
        }



ClmsrGpu::ClmsrGpu( map<int,char*> &repaired_data_, set<int> &aloof_nodes_, map<int, char*> &helper_data_, \
        int repair_blocksize_, map<int,int> &repair_sub_chunks_ind_, ClmsrProfile clmsrProfile_ ) : \
     repaired_data(repaired_data_), aloof_nodes(aloof_nodes_), helper_data(helper_data_),\
     repair_blocksize(repair_blocksize_),repair_sub_chunks_ind(repair_sub_chunks_ind_), clmsrProfile(clmsrProfile_)
     {
        printf("happy should I?");
        //todo: init B_Buf and pin it

        //done: pin memory
        pinMemory(repaired_data, clmsrProfile.chunkSize );
        pinMemory(helper_data, clmsrProfile.subChunkSize*repair_sub_chunks_ind.size() );

        //done: get GpuInfo

        //todo init matrix and put it in the shared memory;
        
     }

ClmsrGpu::~ClmsrGpu()
{
    //todo: free B_buf and unpin it

    //done: unpinmemory
    unpinMemory(repaired_data );
    unpinMemory(helper_data );

}


inline void ClmsrGpu::pinMemory( map<int,char*> map, int size )
{
    for( std::map<int,char*>::iterator iter = map.begin(); iter != map.end(); iter++) {
        //todo:flags
        CUDA_CHECK_RETURN(cudaHostRegister(iter->second, size, cudaHostRegisterPortable));
    }
}

inline void ClmsrGpu::unpinMemory( map<int,char*> map )
{
    for(std::map<int,char*>::iterator iter = map.begin(); iter != map.end(); iter++) {
        CUDA_CHECK_RETURN(cudaHostUnregister(iter->second));
    }
}

DeviceInfo::DeviceInfo()
{
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));
    device = new cudaDeviceProp[deviceCount];
    for( int i = 0; i < deviceCount; i ++ )
    {
        CUDA_CHECK_RETURN(cudaSetDevice(i));
        CUDA_CHECK_RETURN(cudaGetDeviceProperties(&(device[i]),i));
    }
}

DeviceInfo::~DeviceInfo()
{
    free(device);
}

SingleGpuRoute::SingleGpuRoute( int deviceId_, ClmsrGpu* ClmsrGpuP_, int subChunkStart_, int subChunkSize_ ): \
clmsrGpuP(ClmsrGpuP_), deviceId(deviceId_),deviceProp(&((ClmsrGpuP_->deviceInfo).device[deviceId_])),\
subChunkStart(subChunkStart_), subChunkSize(subChunkSize_)
{
    CUDA_CHECK_RETURN(cudaSetDevice(deviceId));
}

void SingleGpuRoute::init()
{
    FT(SingleGpuRoute::init);
}

void SingleGpuRoute::doRepair()
{
    FT(SingleGpuRoute::doRepair);
}
