#include "clmsr_gpu.h"

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include "assert.h"
#include "jerasure_base.h"
#include "gf_base.h"
#include "math.h"
#include <time.h>

extern "C" {
#include "jerasure.h"
}

/*
 ============================================================================
 Name        : cudaTest.cu
 Author      : houyx
 Version     : 0.1
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */


#define talloc(type, num) (type *) malloc(sizeof(type)*(num))
#define A1A2_B1 1
#define A1B2_B1 2
#define B1A2_A1 3
#define A1B1_A2 4
#define B1B2_A1 5
#define GAMMA 6
#define GAMMA_INVERSE 7

#define debughouTab(num, fmt, arg...)     for(int o = 0; o < num; o ++) printf("\t");     printf((const char*)fmt, ##arg)


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
    static const int WORK_SIZE = 65530;
    float *data = new float[WORK_SIZE];

    initialize (data, WORK_SIZE);

    float *recCpu = cpuReciprocal(data, WORK_SIZE);
    float *recGpu = gpuReciprocal(data, WORK_SIZE);
    float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
    float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);


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



ClmsrGpu::ClmsrGpu(  ClmsrProfile clmsrProfile_ ) : \
     clmsrProfile(clmsrProfile_)
     {
        FT(ClmsrGpu::ClmsrGpu);

        //done: init B_Buf and pin it
        CUDA_CHECK_RETURN(cudaHostAlloc(&B_buf, clmsrProfile.q*clmsrProfile.t*sizeof(char*), cudaHostAllocPortable));

        assert(B_buf != NULL);

        for(int i = 0; i < clmsrProfile.q*clmsrProfile.t; i++)
        {
            if(B_buf[i]==NULL)
            {
                CUDA_CHECK_RETURN(cudaHostAlloc(&B_buf[i], (size_t)(clmsrProfile.subChunkSize*clmsrProfile.sub_chunk_no), cudaHostAllocPortable));
                assert(B_buf[i]!= NULL);
            }
        }

        //todo try to put it in the shared memory;
        CUDA_CHECK_RETURN(cudaMalloc(&matrix_gpu, sizeof(int)*clmsrProfile.k*clmsrProfile.m));
        CUDA_CHECK_RETURN(cudaMemcpy(matrix_gpu, clmsrProfile.matrix, sizeof(int)*(clmsrProfile.k*clmsrProfile.m), cudaMemcpyHostToDevice));

     }

int ClmsrGpu::pinAllMemoryForRepair(  map<int,char*>& repaired_data, int sizeRepair,  map<int,char*>& helper_data,  int sizeHelper )
{
    pinMemory(repaired_data, sizeRepair );
    pinMemory(helper_data, sizeHelper );
    return 0;
}

int ClmsrGpu::unpinAllMemoryForRepair(  map<int,char*>& repaired_data,   map<int,char*>& helper_data )
{
    unpinMemory(repaired_data );
    unpinMemory(helper_data );
    return 0;

}

int ClmsrGpu::pinAllMemoryForDecode(  char** data_ptrs, int sizeData,  char** code_ptrs,  int sizeCode )
{   

    //todo: check nu is alloced? may the author is wrong and forget to do it, error!!!
    for( int i = 0; i < clmsrProfile.k + clmsrProfile.nu; i ++ )
    {       //todo:flags
        CUDA_CHECK_RETURN(cudaHostRegister(data_ptrs[i], sizeData, cudaHostAllocPortable));
    }

    for( int i = 0; i < clmsrProfile.m; i ++ )
    {       //todo:flags
        CUDA_CHECK_RETURN(cudaHostRegister(code_ptrs[i], sizeCode, cudaHostAllocPortable));
    }
    return 0;

}

int ClmsrGpu::unpinAllMemoryForDecode(  char** data_ptrs,   char** code_ptrs )
{
    for( int i = 0; i < clmsrProfile.k + clmsrProfile.nu; i ++ )
    {  
        CUDA_CHECK_RETURN(cudaHostUnregister(data_ptrs[i]));
    }

    for( int i = 0; i < clmsrProfile.m; i ++ )
    { 
        CUDA_CHECK_RETURN(cudaHostUnregister(code_ptrs[i]));
    }
    return 0;
}

ClmsrGpu::~ClmsrGpu()
{
    FT(ClmsrGpu::~ClmsrGpu());
    for(int i = 0; i < clmsrProfile.q*clmsrProfile.t; i++)
    {
        CUDA_CHECK_RETURN(cudaFreeHost(B_buf[i]));
    }

    CUDA_CHECK_RETURN(cudaFreeHost(B_buf));

    CUDA_CHECK_RETURN(cudaFree(matrix_gpu));
}


inline void ClmsrGpu::pinMemory( map<int,char*> map, int size )
{
    for( std::map<int,char*>::iterator iter = map.begin(); iter != map.end(); iter++) {
        //todo:flags
        CUDA_CHECK_RETURN(cudaHostRegister(iter->second, size, cudaHostAllocPortable));
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

SingleGpuRoute::SingleGpuRoute( int deviceId_, ClmsrGpu* ClmsrGpuP_, int subSubChunkStart_, int subSubChunkSize_ ): \
clmsrGpuP(ClmsrGpuP_), deviceId(deviceId_),deviceProp(((ClmsrGpuP_->deviceInfo).device[deviceId_])),\
subSubChunkStart(subSubChunkStart_), subSubChunkSize(subSubChunkSize_), clmsrProfileP(&(ClmsrGpuP_->clmsrProfile))\
{
    CUDA_CHECK_RETURN(cudaSetDevice(deviceId));

    //todo new: delete
    pieceKernelGridSize = deviceProp.multiProcessorCount;
    pieceKernelBlockSize = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    planeKernelGridSize = deviceProp.multiProcessorCount;
    planeKernelBlockSize = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);


    int layerSize = subSubChunkSize * clmsrProfileP->q * clmsrProfileP->t;
    //todo: check if overflowed: now because layer 1's A1 is need, B1 is caled, couple A2 is need, B2 is caled, so at most is 4 times memory.
    if(float(deviceProp.totalGlobalMem) < float(layerSize) * 4.0 + clmsrProfileP->k*clmsrProfileP->m*sizeof(int) )
    {
        pieceCount = layerSize*4/deviceProp.totalGlobalMem + 1;
    }
    else
    {
        pieceCount = 1;
    }

    init();

    CUDA_CHECK_RETURN( cudaMalloc(& gf_table.g_log, GF_FIELD_SIZE_8_GPU) );
    CUDA_CHECK_RETURN( cudaMalloc(& gf_table.g_anti_log, GF_FIELD_SIZE_8_GPU*2) );
    CUDA_CHECK_RETURN( cudaMalloc(& gf_table.g_inv, GF_FIELD_SIZE_8_GPU) );

}

SingleGpuRoute::~SingleGpuRoute()
{
    deinit();
    CUDA_CHECK_RETURN( cudaFree(gf_table.g_log) );
    CUDA_CHECK_RETURN( cudaFree(gf_table.g_anti_log) );
    CUDA_CHECK_RETURN( cudaFree(gf_table.g_inv) );
}


int SingleGpuRoute::__getPieceSize( int i )
{
    //error!!!!
    //todo: think about thread
    int baseSize = subSubChunkSize/pieceCount;
    //int baseSize = subSubChunkSize%pieceCount == 0;


    if( subSubChunkSize%pieceCount == 0 )
    {
        return baseSize;
    }
    else if( (i + 1) <= subSubChunkSize%pieceCount )
    {
        baseSize ++;
    }

    return baseSize;
}

void SingleGpuRoute::init()
{
    FT(SingleGpuRoute::init);
    int qt = clmsrProfileP->q*clmsrProfileP->t;
    streams = new cudaStream_t[qt];
    events = new cudaEvent_t[clmsrProfileP->q*clmsrProfileP->t];

    for( int i = 0; i < qt; i ++ )
    {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
    }

    for( int i = 0; i < qt; i ++ )
    {
        CUDA_CHECK_RETURN(cudaEventCreate(&events[i]));
    }
    //cudaEventRecord(events[i], streams[i]);
    //todo: init streams and events
}


inline void get_plane_vector(int q, int t, int z, int* z_vec)
{
  int i ;

  for(i = 0; i<t; i++ ){
    z_vec[t-1-i] = z%q;
    z = (z - z_vec[t-1-i])/q;
  }
  return;
}

__global__ void testGf( gf_w8_log_gpu gf_table, int w )
{
    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    set_gf_table(gf_table);

    if( idx == 0 )
    {
        printf("=========In testGf=============mul\n");
        for( int i = 0; i < 256; i ++ )
        {
            for( int j = 0; j < 256; j ++ )
            {
                printf( "%d\t*\t%d\t=\t%d\n", i, j, galois_single_multiply_gpu_logtable_w8((unsigned char)i, (unsigned char)j) );
            }
        }
        printf("=========In testGf=============div\n");
        for( int i = 0; i < 256; i ++ )
        {
            for( int j = 1; j < 256; j++ )
            {
                printf( "%d\t/\t%d\t=\t%d\n", i, j, galois_single_divide_gpu_logtable_w8((unsigned char)i, (unsigned char)j) );
            }
        }
    }
}

__global__ void pieceKernelGamma( gf_w8_log_gpu gf_table, unsigned char gamma, unsigned char** dataPt,  int nodeId, int calType, int dataSize, int patchSize, int w )
{
    set_gf_table(gf_table);

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    unsigned char tmatrix[4];


    unsigned char* A[2];
    unsigned char* dest[2];

    switch(calType)
    {
        case GAMMA :

            tmatrix[0] = 1;
            tmatrix[1] = gamma;
            tmatrix[2] = gamma;
            tmatrix[3] = 1;

            A[0] = dataPt[nodeId] + patchSize*0; //B2
            A[1] = dataPt[nodeId] + patchSize*1; //B1

            dest[0] = dataPt[nodeId] + patchSize*2;
            dest[1] = dataPt[nodeId] + patchSize*3;

            break;

        case GAMMA_INVERSE  :

            tmatrix[0] = galois_single_divide_gpu_logtable_w8(1, 1 ^ (galois_single_multiply_gpu_logtable_w8(gamma, gamma)));
            tmatrix[1] = galois_single_multiply_gpu_logtable_w8(gamma,galois_single_divide_gpu_logtable_w8(1, 1 ^ (galois_single_multiply_gpu_logtable_w8(gamma, gamma))));
            tmatrix[2] = tmatrix[1];
            tmatrix[3] = tmatrix[0];



            A[0] = dataPt[nodeId] + patchSize*2; //B2
            A[1] = dataPt[nodeId] + patchSize*3; //B1

            dest[0] = dataPt[nodeId] + patchSize*0;
            dest[1] = dataPt[nodeId] + patchSize*1;

            break;

        default:
            printf("error Case %d\n", calType);
            __threadfence_system();
            asm("trap;");
            break;
    }

    for( int i = idx; i < dataSize; i += threadsNum )
    {
        //mark = 0;
        //todo: optmize a*1 case;
        dest[0][i] = galois_single_multiply_gpu_logtable_w8(A[0][i], tmatrix[0]) ^ galois_single_multiply_gpu_logtable_w8(A[1][i], tmatrix[1]);
        dest[1][i] = galois_single_multiply_gpu_logtable_w8(A[0][i], tmatrix[2]) ^ galois_single_multiply_gpu_logtable_w8(A[1][i], tmatrix[3]);
    }
}

__global__ void pieceKernel( gf_w8_log_gpu gf_table, unsigned char gamma, unsigned char** dataPt,  int nodeId, int calType, int dataSize, int patchSize, int w )
{


    set_gf_table(gf_table);

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;


    unsigned char tmatrix[2];
    unsigned char* in_dot[2];
    unsigned char* dest;
    //todo: make the preceed all the same by changing the pointer and matrix;


    switch(calType)
    {
        case A1A2_B1        :


            tmatrix[0] = 1;
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId];  //A1
            in_dot[1] = dataPt[nodeId] + patchSize; //A2

            dest = dataPt[nodeId] + patchSize*2; //B1
            break;

        case A1B2_B1        :

            tmatrix[0] = (1 ^ galois_single_multiply_gpu_logtable_w8(gamma, gamma));
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId];  //A1
            in_dot[1] = dataPt[nodeId] + patchSize*3; //B2

            dest = dataPt[nodeId] + patchSize*2; //B1
            break;

        case B1A2_A1        :

            tmatrix[0] = 1;
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId] + patchSize*2;  //B1
            in_dot[1] = dataPt[nodeId] + patchSize*1; //A2

            dest = dataPt[nodeId]; //A1

            break;

        case A1B1_A2        :

            tmatrix[0] = galois_single_divide_gpu_logtable_w8(1,gamma);
            tmatrix[1] = tmatrix[0];

            in_dot[0] = dataPt[nodeId]+ patchSize*2;  //B1
            in_dot[1] = dataPt[nodeId]; //A1

            dest = dataPt[nodeId] + patchSize; //A2

            break;       


        case B1B2_A1        :

            tmatrix[0] = galois_single_divide_gpu_logtable_w8(1, 1 ^ (galois_single_multiply_gpu_logtable_w8(gamma, gamma)));
            tmatrix[1] = galois_single_multiply_gpu_logtable_w8(gamma,galois_single_divide_gpu_logtable_w8(1, 1 ^ (galois_single_multiply_gpu_logtable_w8(gamma, gamma))));
            

            in_dot[0] = dataPt[nodeId] + patchSize*2;  //B1
            in_dot[1] = dataPt[nodeId] + patchSize*3; //B2

            dest = dataPt[nodeId]; //A1
            break;

        default:
            
            printf("error Case %d\n", calType);
            __threadfence_system();
            asm("trap;");
            break;
    }
    
/*    __threadfence_system();
    __syncthreads();
    debughouTab(1, "ready for loooooop %d\n", idx);
    __syncthreads();
    __threadfence_system();*/

    for( int i = idx; i < dataSize; i += threadsNum )
    {
        //todo: optmize a*1 case;
        dest[i] = galois_single_multiply_gpu_logtable_w8(in_dot[0][i], tmatrix[0]) ^ galois_single_multiply_gpu_logtable_w8(in_dot[1][i], tmatrix[1]);    
    }
}


__global__ void planeKernel(gf_w8_log_gpu gf_table,  const int k, int nu, const int m, int w, int q, int t, int* matrix, int* decode_matrix, \
    int* dm_ids, int* erasure_loc_data, int erased_data_size, int* erasure_loc_coding, int erased_coding_size, unsigned char** dataPt, int dataSize, int patchSize )
{
    set_gf_table( gf_table );

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    //load matrix to gpu shared memery;
    extern __shared__ unsigned char s[];
    unsigned char *sh_decode_matrix = s;                        //
    unsigned char *sh_matrix = (unsigned char*)&sh_decode_matrix[k*k]; //

    for( int t = idx; t < k*k; t += threadsNum )
    {
        sh_decode_matrix[t] = (unsigned char) decode_matrix[t];
    }

    //todo: optmize and merge
    for( int t = idx; t < k*m; t += threadsNum )
    {
        sh_matrix[t] = (unsigned char) matrix[t];
    }
    
    __syncthreads();

    // //debug new
    // if( idx == threadsNum - 1 )
    // {
    //     printf(">>>>>>>> in plane, matrix:\n");
    //     for( int i = 0; i < k*m; i ++ )
    //     {
    //         printf("%d: %d\t", i, sh_matrix[i] );
    //     }
    //     printf("\n decoding_matrix: \n");
    //     for( int i = 0; i < k*k; i ++ )
    //     {
    //         printf("%d: %d\t", i, sh_decode_matrix[i] );
    //     }
    //     printf("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
    //     printf("\n erased_data_size: %d erased_coding_size: %d\n", erased_data_size, erased_coding_size);
    // }

    unsigned char * data_now;
    unsigned char * erasure_now;

    for( int i = 0 ; i < k; i ++ )
    {
        data_now = dataPt[dm_ids[i]] + patchSize*2;//B1 is what we want do RS

        for( int j = 0; j < erased_data_size; j ++ )
        {
            erasure_now = dataPt[erasure_loc_data[j]] + patchSize*2;
            
            for( int t = idx; t < dataSize; t += threadsNum )
            {
                if( i == 0 )
                {
                    //todo new: get the fuck shared memory working right
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], (decode_matrix[j*k + i]));
                }
                else
                {
                    erasure_now[t] = erasure_now[t]^galois_single_multiply_gpu_logtable_w8( data_now[t], (decode_matrix[j*k + i]));
                }    
            }
        }
    }

    for( int i = 0 ; i < k; i ++ )
    {
        data_now = dataPt[i] + patchSize*2;//B1 is what we want do RS

        for( int j = 0; j < erased_coding_size; j ++ )
        {
            erasure_now = dataPt[erasure_loc_coding[j]] + patchSize*2;
            
            for( int t = idx; t < dataSize; t += threadsNum )
            {
                //todo: ensure that dest[i] is all clean first time;
                if( i == 0 )
                {
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], ((matrix[j*k + i])));
                }
                else
                {
                    erasure_now[t] = erasure_now[t]^galois_single_multiply_gpu_logtable_w8( data_now[t], ((matrix[j*k + i])));
                }      
                //debug new
                // if( i == 1 ){
                //     printf( "node[%d][%d]: %d = node[%d][%d]: %d * matrix_sh[%d][%d] : %d,  matrix[][]: %d, ----k: %d\n", j, t, erasure_now[t], i, t, data_now[t], j, i, sh_matrix[1], (unsigned char)matrix[1], k );
                // }    
            }
        }
    }
}

void debugPrintVecGpu( unsigned char* dataGpu, int size, int node, string name )
{
    cout << "node:\t" << node << "\t" << name << "=====================" << endl << endl;
    unsigned char data[size];

    CUDA_CHECK_RETURN( cudaMemcpy( data, dataGpu, size, cudaMemcpyDeviceToHost) );
    
    for( int i = 0; i < size; i ++ )
    {
        printf("%u=%c,", (unsigned char) data[i],data[i]);
    }

    cout << "==================================================" << endl;
    
}

void debugPrintVec( char* data, int size, int node, string name )
{
    cout << "node:\t" << node << "\t" << name << "=====================" << endl << endl;
    
    for( int i = 0; i < size; i ++ )
    {
        printf("%u,", (unsigned char) data[i]);
    }

    cout << "\n\n==================================================" << endl;
    
}


int SingleGpuRoute::doRepair( map<int,char*> &repaired_data, set<int> &aloof_nodes,
    map<int, char*> &helper_data, int repair_blocksize, map<int,int> &repair_sub_chunks_ind, char** B_buf )
{
    FT(SingleGpuRoute::doRepair);


    init_gf_log_w8_gpu();

    const int k = clmsrProfileP->k + clmsrProfileP->nu;
    const int m = clmsrProfileP->m;

    const int sub_chunksize = clmsrProfileP->subChunkSize;
    const int q = clmsrProfileP->q, t = clmsrProfileP->t;
    const int qt = q * t;
    int* z_vec;
    

    CUDA_CHECK_RETURN( cudaHostAlloc(&z_vec, t * sizeof(int), cudaHostAllocPortable) );

    map<int, set<int> > ordered_planes;
    map<int, int> repair_plane_to_ind;
    int order = 0;
    int x,y, node_xy, node_sw, z_sw;
    char *A1, *A2, *B1, *B2;
    int count_retrieved_sub_chunks = 0;
    int num_erased = 0;

    int *decode_matrix_gpu;
    int *decode_matrix;
    int *dm_ids;
    int *dm_ids_gpu;
    int *erased;
    int *erasure_loc_data;
    int *erasure_loc_data_gpu;
    int *erasure_loc_coding;
    int *erasure_loc_coding_gpu;
    int erased_data_size = 0;
    int erased_coding_size = 0;

    bool init_matrix = false;

    dm_ids = talloc(int, k);
    decode_matrix = talloc(int, k*k);
    erased = talloc(int, k + m);
    erasure_loc_data = talloc(int, qt);
    erasure_loc_coding = talloc(int, qt);

    CUDA_CHECK_RETURN(cudaMalloc(&decode_matrix_gpu, k * k *sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&dm_ids_gpu, k * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_data_gpu, qt * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_coding_gpu, qt * sizeof(int)));

    //malloc the data temp space on the gpu a piece has 4 things: A1,A2,B1,B2;
    int pieceSizeMax = __getPieceSize(0);
    unsigned char* planeOnGpu[qt];
    for( int i = 0; i < qt; i ++ )
    {
    CUDA_CHECK_RETURN(cudaMalloc(&planeOnGpu[i], pieceSizeMax*4));
    }

    unsigned char** planePOnGpuK;

    CUDA_CHECK_RETURN(cudaMalloc(&planePOnGpuK, qt*sizeof(unsigned char*)));

    CUDA_CHECK_RETURN(cudaMemcpy(planePOnGpuK, planeOnGpu, qt*sizeof(unsigned char*), cudaMemcpyHostToDevice));

    int plane_count = 0;
    int erasure_locations[qt];

    //get order of all planes
    for(map<int,int>::iterator i = repair_sub_chunks_ind.begin(); i != repair_sub_chunks_ind.end(); ++i)
    {
    get_plane_vector(q,  t, i->second, z_vec);
    order = 0;
    //check across all erasures
    for(map<int,char*>::iterator j = repaired_data.begin(); j != repaired_data.end(); ++j)
    {
    if(j->first% q == z_vec[j->first/q])order++;
    }
    assert(order>0);
    ordered_planes[order].insert(i->second);
    repair_plane_to_ind[i->second] = i->first;
    }



    //repair planes in order
    for(order=1; ;order++){
    if(ordered_planes.find(order) == ordered_planes.end())
    {
    break;
    }
    else
    {
    plane_count += ordered_planes[order].size();

    //where GPU works
    for(set<int>::iterator z=ordered_planes[order].begin(); z != ordered_planes[order].end(); ++z)
    {
    get_plane_vector(q,t,*z, z_vec);

    int pieceOffset = subSubChunkStart;
    int pieceSize = 0;
    init_matrix = false;


    for( int pi = 0; pi < pieceCount; pi ++ )
    {
    pieceSize = __getPieceSize(pi);

    num_erased = 0;
    for(y=0; y < t; y++)
    {
    for(x = 0; x < q; x++)
    {
        node_xy = y*q + x;//todo: check pow is right
        z_sw = (*z) + (x - z_vec[y])*(int)(pow(q,t-1-y));
        node_sw = y*q + z_vec[y];

        if( (repaired_data.find(node_xy) != repaired_data.end()) )
        {//case of erasure, aloof node can't get a B.

            erasure_locations[num_erased] = node_xy;
            num_erased++;

            if( repaired_data.find(node_sw) != repaired_data.end() )
            {//node_sw must be a helper
                if(x < z_vec[y])//todo: check if is right
                {
                    B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*3,  B2, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                }
            }
            else// check new, if necessary
            {
                assert( helper_data.find(node_sw) != helper_data.end() );
                A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];

                if( z_vec[y] != x)
                {
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*1, A2, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                    //get_B1_fromA1A2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, A2, sub_chunksize);
                }
            }
        }
        else if( (aloof_nodes.find(node_xy) != aloof_nodes.end()) )
        {
            erasure_locations[num_erased] = node_xy;
            num_erased++;
        }
        else
        {//should be in helper data
            assert(helper_data.find(node_xy) != helper_data.end());
            //so A1 is available, need to check if A2 is available.
            A1 = &helper_data[node_xy][repair_plane_to_ind[*z]*sub_chunksize + pieceOffset];
            
            // later will do get_A2_from_B1_A1, consider this as an erasure, if A2 not found.
            if(repair_plane_to_ind.find(z_sw) == repair_plane_to_ind.end())
            {
                erasure_locations[num_erased] = node_xy;
                CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy]   , A1, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                num_erased++;
            }
            else
            {
                if(repaired_data.find(node_sw) != repaired_data.end())
                {
                    //todo: 尝试在这里加assert证明repaired_data一定被算出来了，解决上面的todo
                    assert(z_sw < sub_chunk_no);
                    A2 = &repaired_data[node_sw][z_sw*sub_chunksize + pieceOffset];
                    //transfer A1A2 and cal B1 B2;
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                A1, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax, A2, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                    
                    //todo new: get right stream
                    pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table,  clmsrProfileP->gamma,planePOnGpuK,node_xy, A1A2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                }
                else if(aloof_nodes.find(node_sw) != aloof_nodes.end())
                {
                    B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                  A1, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                    CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*3, B2, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                    
                    //todo: find a parameter
                    pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,node_xy,A1B2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                }
                else
                {
                    assert(helper_data.find(node_sw) != helper_data.end());
    //dout(10) << "obtaining B1 from A1 A2 for node: " << node_xy << " on plane:" << *z << dendl;
                    A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                    
                    if( z_vec[y] != x)
                    {
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                  A1, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*1, A2, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                        //todo: find a parameter
                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,node_xy,A1A2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                    }
                    else
                    {
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*2, A1, pieceSize, cudaMemcpyHostToDevice,streams[node_xy]) );
                    }
                }
            }

        }


    }//y
    }//x

    erasure_locations[num_erased] = -1;
    assert(num_erased <= m);

    if(!init_matrix)
    {
    init_matrix = true;
    
    if(erasures_to_erased_gpu (k, m, erasure_locations, erased ) < 0 )
    {
        printf("haha, you get an error when calling erasures_to_erased_gpu!\n");
        return -1;
    }
    
    if (jerasure_make_decoding_matrix(k, m, clmsrProfileP->w, clmsrProfileP->matrix, erased, decode_matrix, dm_ids) < 0) 
    {
    printf("Can not get decoding matrix!!!\n");
    return -1;
    }

    CUDA_CHECK_RETURN( cudaMemcpy(decode_matrix_gpu, decode_matrix, k*k*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(dm_ids_gpu, dm_ids, k*sizeof(int), cudaMemcpyHostToDevice) );
    
    erased_data_size = full_erased_list_data( k, m, erasure_loc_data, erased );
    erased_coding_size = full_erased_list_coding( k, m, erasure_loc_coding, erased );

    CUDA_CHECK_RETURN( cudaMemcpy(erasure_loc_data_gpu, erasure_loc_data, qt*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(erasure_loc_coding_gpu, erasure_loc_coding, qt*sizeof(int), cudaMemcpyHostToDevice) );

    }

    //todo new: find a stream
    planeKernel<<<planeKernelGridSize,planeKernelBlockSize,(k*k + k * m)*sizeof(char)>>>(gf_table, \
    k, clmsrProfileP->nu, m, clmsrProfileP->w, q, t,\
    clmsrGpuP->matrix_gpu, decode_matrix_gpu, dm_ids_gpu, erasure_loc_data_gpu, erased_data_size, erasure_loc_coding_gpu, erased_coding_size, planePOnGpuK, pieceSize, pieceSizeMax\
    );

    for(int i = 0; i < num_erased; i++)
    {
    x = erasure_locations[i]%q;
    y = erasure_locations[i]/q;
    node_sw = y*q+z_vec[y];
    z_sw = (*z) + (x - z_vec[y]) * (int)pow(q,t-1-y);

    //make sure it is not an aloof node before you retrieve repaired_data
    if( aloof_nodes.find(erasure_locations[i]) == aloof_nodes.end())
    {
        if(x == z_vec[y] )
        {//hole-dot pair (type 0)
            A1 = &repaired_data[erasure_locations[i]][*z*sub_chunksize + pieceOffset];
            CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost, streams[erasure_locations[i]]) );
            count_retrieved_sub_chunks++;
        }//can recover next case (type 2) only after obtaining B's for all the planes with same order
        else
        {
            //恢复的是个want_to_read节点
            if(repaired_data.find(erasure_locations[i]) != repaired_data.end() )
            {//this is a hole (lost node)
                A1 = &repaired_data[erasure_locations[i]][*z*sub_chunksize + pieceOffset];
                //check if type-2
                //node_sw也是一个want_to_read节点
                if( repaired_data.find(node_sw) != repaired_data.end())
                {
                    if(x < z_vec[y])//当满足这个条件时证明另一个B2已经算出来了,需要在上面传入
                    {//recover both A1 and A2 here
                        A2 = &repaired_data[node_sw][z_sw*sub_chunksize + pieceOffset];

                        pieceKernelGamma<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[1]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],GAMMA_INVERSE, pieceSize, pieceSizeMax, clmsrProfileP->w);
                        //debug
                        //CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]],                  pieceSize, cudaMemcpyDeviceToHost, streams[erasure_locations[i]]) );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync( A2, planeOnGpu[erasure_locations[i]] + pieceSizeMax*1, pieceSize, cudaMemcpyDeviceToHost, streams[erasure_locations[i]]) );
                        count_retrieved_sub_chunks = count_retrieved_sub_chunks + 2;
                    }
                    else//把算好的B1拿出来
                    {
                        B1 = &B_buf[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize + pieceOffset];
                        CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost, streams[erasure_locations[i]]) );
                    }
                }
                else//node_sw是一个helper节点：　这里不可能是一个aloof节点，因为同一个y-section的都是helper节点．
                {
                    assert(helper_data.find(node_sw) != helper_data.end());
                    assert(repair_plane_to_ind.find(z_sw) !=  repair_plane_to_ind.end());

                    pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[erasure_locations[i]]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],B1A2_A1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                    //debug
                    //CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
                    CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]], pieceSize, cudaMemcpyDeviceToHost, streams[erasure_locations[i]]) );
                    count_retrieved_sub_chunks++;
                }
            }
            else
            {
                //这里是说我是一个helper,但我的z_sw是一个want_to_read,所以我需要恢复A2;
                //not a hole and has an erasure in the y-crossection.
                assert(repaired_data.find(node_sw) != repaired_data.end());
                if(repair_plane_to_ind.find(z_sw) == repair_plane_to_ind.end())
                {
                    A2 = &repaired_data[node_sw][z_sw*sub_chunksize + pieceOffset];
                    
                    pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[erasure_locations[i]]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],A1B1_A2, pieceSize, pieceSizeMax, clmsrProfileP->w);
                    //debug
                    //CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
                    CUDA_CHECK_RETURN( cudaMemcpyAsync( A2, planeOnGpu[erasure_locations[i]] + pieceSizeMax*1, pieceSize, cudaMemcpyDeviceToHost,streams[erasure_locations[i]]) );

                    count_retrieved_sub_chunks++;
                }
            }
        }//type-1 erasure recovered.
    }//not an aloof node
    else
    {
        B1 = &B_buf[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize + pieceOffset];
        CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost,streams[erasure_locations[i]]) );
    }
    }//erasures
    pieceOffset += pieceSize;
    }
    }//planes of a particular order

    }
    }
    assert(repair_sub_chunks_ind.size() == (unsigned)plane_count);
    assert(sub_chunk_no*repaired_data.size() == (unsigned)count_retrieved_sub_chunks);



    CUDA_CHECK_RETURN(cudaFreeHost(z_vec));
    CUDA_CHECK_RETURN(cudaFree(decode_matrix_gpu));
    CUDA_CHECK_RETURN(cudaFree(dm_ids_gpu));
    CUDA_CHECK_RETURN(cudaFree(erasure_loc_data_gpu));
    CUDA_CHECK_RETURN(cudaFree(erasure_loc_coding_gpu));


    for( int i = 0; i < qt; i ++ )
    {
    CUDA_CHECK_RETURN(cudaFree(planeOnGpu[i]));
    }

    CUDA_CHECK_RETURN(cudaFree(planePOnGpuK));


    return 0;
}


int is_erasure_type_1(int m, int ind, erasure_t_gpu* erasures, int* z_vec){

  // Need to look for the column of where erasures[i] is and search to see if there is a hole dot pair.
  int i;

  if(erasures[ind].x == z_vec[erasures[ind].y]) return 0; //type-0 erasure

  for(i=0; i < m; i++){
    if(erasures[i].y == erasures[ind].y){
      if(erasures[i].x == z_vec[erasures[i].y]){
    return 0;
      }
    }
  }
  return 1;

}

void get_erasure_coordinates_gpu(int m, int q, int t, int* erasure_locations, erasure_t_gpu* erasures )
{
  int i;

  for(i = 0; i<m; i++){
    if(erasure_locations[i]==-1)break;
    erasures[i].x = erasure_locations[i]%q;
    erasures[i].y = erasure_locations[i]/q;
  }
}

int SingleGpuRoute::doDecode \
( int* erasure_locations, char** data_ptrs, char** code_ptrs, int* erased, \
                            int num_erasures, int* order, int* weight_vec, int max_weight, int size)

{

    char** B_buf = clmsrGpuP->B_buf;

    FT(SingleGpuRoute::doDecode);
//debug
//     printf("******************\ngamma:\t%d\nq:\t%d\nt:\t%d\nd:\t%d\nsize: %d\n****************haha\n", clmsrProfileP->gamma,clmsrProfileP->q,clmsrProfileP->t,clmsrProfileP->d,size );
//   printf("check in doDecode>>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
//   for( int j = 0; j < clmsrProfileP->k; j ++ )
//   {
//     printf("j: %d----------------\n", j
//       );
//     for( int i = 0; i < size; i ++ )
//     {
//         printf("%c,", data_ptrs[j][i] );
//     }
//     printf("----------------------------------------------------------------\n");
//   }
//   printf("check in doDecode>>>>>>>>>>>>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");




    init_gf_log_w8_gpu();

    erasure_t_gpu erasures[clmsrProfileP->m];
    get_erasure_coordinates_gpu(clmsrProfileP->m, clmsrProfileP->q, clmsrProfileP->t, erasure_locations, erasures);

    int i;

    char *A1 = NULL, *A2 = NULL;

    int hm_w;

    assert(size%clmsrProfileP->subChunkSize == 0);
    int ss_size = clmsrProfileP->subChunkSize;

//debug
   printf(" k: %d\t nu: %d\t m: %d\n",clmsrProfileP->k, clmsrProfileP->nu, clmsrProfileP->m );
    const int k = clmsrProfileP->k + clmsrProfileP->nu;
    const int m = clmsrProfileP->m;

    const int sub_chunksize = clmsrProfileP->subChunkSize;

    const int q = clmsrProfileP->q, t = clmsrProfileP->t, nu = clmsrProfileP->nu;
    const int qt = q * t;
    int* z_vec;

    CUDA_CHECK_RETURN( cudaHostAlloc(&z_vec, t * sizeof(int), cudaHostAllocPortable) );


    map<int, set<int> > ordered_planes;
    map<int, int> repair_plane_to_ind;
    int z, x,y, node_xy, node_sw, z_sw;
    int *decode_matrix_gpu;
    int *decode_matrix;
    int *dm_ids;
    int *dm_ids_gpu;
    int *erasure_loc_data;
    int *erasure_loc_data_gpu;
    int *erasure_loc_coding;
    int *erasure_loc_coding_gpu;
    int erased_data_size = 0;
    int erased_coding_size = 0;

    dm_ids = talloc(int, k);
    decode_matrix = talloc(int, k*k);
    erasure_loc_data = talloc(int, qt);
    erasure_loc_coding = talloc(int, qt);

    CUDA_CHECK_RETURN(cudaMalloc(&decode_matrix_gpu, k * k *sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&dm_ids_gpu, k * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_data_gpu, qt * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_coding_gpu, qt * sizeof(int)));

    //malloc the data temp space on the gpu a piece has 4 things: A1,A2,B1,B2;
    
    int pieceSizeMax = __getPieceSize(0);
    unsigned char* planeOnGpu[qt];
    for( int i = 0; i < qt; i ++ )
    {
        CUDA_CHECK_RETURN(cudaMalloc(&planeOnGpu[i], pieceSizeMax*4));
    }

    unsigned char** planePOnGpuK;

    CUDA_CHECK_RETURN(cudaMalloc(&planePOnGpuK, qt*sizeof(unsigned char*)));

    CUDA_CHECK_RETURN(cudaMemcpy(planePOnGpuK, planeOnGpu, qt*sizeof(unsigned char*), cudaMemcpyHostToDevice));

    //debug
    //printf("test here B_Buf: %p B_Buf[0]: %p\n", B_buf, B_buf[0]);
    /*char* ttemp;
    CUDA_CHECK_RETURN(cudaMalloc(&ttemp, 1000*sizeof(char)));
    CUDA_CHECK_RETURN( cudaMemcpy( B_buf[0], ttemp, 9, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN(cudaFree(ttemp));*/
    //init decode vars: etc: decode_matrix
    printf("get before jerasure_make_decoding_matrix=========================================\n");
    if (jerasure_make_decoding_matrix(k, m, clmsrProfileP->w, clmsrProfileP->matrix, erased, decode_matrix, dm_ids) < 0) 
    {
      printf("Can not get decoding matrix!!!\n");
      return -1;
    }

    CUDA_CHECK_RETURN( cudaMemcpy(decode_matrix_gpu, decode_matrix, k*k*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(dm_ids_gpu, dm_ids, k*sizeof(int), cudaMemcpyHostToDevice) );
    

    erased_data_size = full_erased_list_data( k, m, erasure_loc_data, erased );
    erased_coding_size = full_erased_list_coding( k, m, erasure_loc_coding, erased );

    CUDA_CHECK_RETURN( cudaMemcpy(erasure_loc_data_gpu, erasure_loc_data, qt*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(erasure_loc_coding_gpu, erasure_loc_coding, qt*sizeof(int), cudaMemcpyHostToDevice) );
    printf("get afer jerasure_make_decoding_matrix=========================================\n");


    //debug
    /*unsigned char test[pieceSizeMax];

    for( int i = 0; i < pieceSizeMax; i++ )
    {
        test[i] = (i%255);
    }

    CUDA_CHECK_RETURN( cudaMemcpy(planeOnGpu[0],                test, __getPieceSize(0), cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy(planeOnGpu[0] + pieceSizeMax, test, __getPieceSize(0), cudaMemcpyHostToDevice) );
    
    unsigned char a1g[pieceSizeMax],a2g[pieceSizeMax];
    CUDA_CHECK_RETURN( cudaMemcpy( a1g, planeOnGpu[0] + pieceSizeMax*0, __getPieceSize(0), cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaMemcpy( a2g, planeOnGpu[0] + pieceSizeMax*1, __getPieceSize(0), cudaMemcpyDeviceToHost) );
      

    printf("\na1g--------------------------------------------------------------\n");
    for( int i = 0; i < ss_size; i ++ )
    {
        printf("%u,", (unsigned char) a1g[i] );
    }

    printf("\na2g--------------------------------------------------------------\n");
    for( int i = 0; i < ss_size; i ++ )
    {
        printf("%u,", (unsigned char) a2g[i] );
    }
    printf("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n");
    return 0;    */                         

    timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for(hm_w = 0; hm_w <= max_weight; hm_w++)
    {
            int pieceOffset = subSubChunkStart;
            int pieceSize = 0;
            //init_matrix = false;

                //printf("get set<int>::iterator z=ordered_planes[order].begin(); z != ordered_planes[order].end()=========================================\n");

                //printf("pieceCount %d \n", pieceCount );

        for( int pi = 0; pi < pieceCount; pi ++ )
        {
            pieceSize = __getPieceSize(pi);

            for(z = 0; z< clmsrProfileP->sub_chunk_no; z++)
            {
                //cout << ">>>>> z: " << z << endl;
                if(order[z]==hm_w)
                {
                    get_plane_vector(q, t, z,z_vec);
                    //__decode_erasures(erasure_locations, z, z_vec, data_ptrs, code_ptrs, ss_size, B_buf, pi, pieceSize, pieceSizeMax, );
                    for(x=0; x < q; x++)
                    {
                        for(y=0; y<t; y++)
                        {
                            //todo: may not need because A1 is erasured error!!!
                            /*if( erased[y*q+x] != 1 )
                            {*/
                                node_xy = y*q+x; 
                                node_sw = y*q+z_vec[y];
                                z_sw = z + (x - z_vec[y]) * (int)pow(q,t-1-y);


                                
                                A1 = (node_xy < k+nu) ? &data_ptrs[node_xy][z*ss_size + pieceOffset] : &code_ptrs[node_xy-k-nu][z*ss_size + pieceOffset];
                                A2 = (node_sw < k+nu) ? &data_ptrs[node_sw][z_sw*ss_size + pieceOffset] : &code_ptrs[node_sw-k-nu][z_sw*ss_size + pieceOffset];

                                if(erased[node_xy] == 0)
                                { //if not an erasure 
                                    if(z_vec[y] != x)
                                    {//not a dot
                                        //get_B1_fromA1A2(&B_buf[node_xy][z*ss_size + pieceOffset], A1, A2, ss_size);
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                A1, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax, A2, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                                        

                                        //todo: find a parameter
                                        //debug
                                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table, clmsrProfileP->gamma, planePOnGpuK, node_xy, A1A2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w );
                                        //cudaDeviceSynchronize();
                                    }
                                    else
                                    { //dot
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*2, A1, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                                        //memcpy(&B_buf[node_xy][z*ss_size  + pieceOffset], A1, ss_size);
                                    }
                                }
                            //}
                        }
                    }

                    //debug
                //     for(x=0; x < q; x++){
                //       for(y=0; y<t; y++){
                //           node_xy = y*q+x; 
                //           node_sw = y*q+z_vec[y];
                //           z_sw = z + (x - z_vec[y]) * (int)pow(q,t-1-y);


                //         if(erased[node_xy] == 0)
                //                 { //if not an erasure 
                //                     if(z_vec[y] != x)
                //                     {

                //           char* B1 = &B_buf[node_xy][z*pieceSize];
                //           A1 = (node_xy < k+nu) ? &data_ptrs[node_xy][z*ss_size + pieceOffset] : &code_ptrs[node_xy-k-nu][z*ss_size + pieceOffset];
                //           A2 = (node_sw < k+nu) ? &data_ptrs[node_sw][z_sw*ss_size + pieceOffset] : &code_ptrs[node_sw-k-nu][z_sw*ss_size + pieceOffset];



                          


                //           printf("\n\nGPU===================================node_xy: %d data( size %d ):\n\n", node_xy,pieceSize );
                //           for( int i = 0; i < ss_size; i ++ )
                //           {
                //             printf("%u,", (unsigned char) A1[i] );
                //           }
                //           printf("\nA2--------------------------------------------------------------\n");
                //           for( int i = 0; i < ss_size; i ++ )
                //           {
                //             printf("%u,", (unsigned char) A2[i] );
                //           }
                          

                //           char a1g[pieceSize],a2g[pieceSize];
                //           CUDA_CHECK_RETURN( cudaMemcpyAsync( a1g, planeOnGpu[node_xy] + pieceSizeMax*0, pieceSize, cudaMemcpyDeviceToHost) );
                //           CUDA_CHECK_RETURN( cudaMemcpyAsync( a2g, planeOnGpu[node_xy] + pieceSizeMax*1, pieceSize, cudaMemcpyDeviceToHost) );
                          
                //           CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[node_xy] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost) );
                          
                //           printf("\nA1G--------------------------------------------------------------\n");
                //           for( int i = 0; i < ss_size; i ++ )
                //           {
                //             printf("%u,", (unsigned char) a1g[i] );
                //           }
                //           printf("\nA2G--------------------------------------------------------------\n");
                //           for( int i = 0; i < ss_size; i ++ )
                //           {
                //             printf("%u,", (unsigned char) a2g[i] );
                //           }

                //           printf("\nB1--------------------------------------------------------------\n");
                //           for( int i = 0; i < ss_size; i ++ )
                //           {
                //             printf("%u,", (unsigned char) B1[i] );
                //           }
                //           printf("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n");
                //         }
                //     }
                // }
                //       }

                    //Decode in B's
                   /* jerasure_matrix_decode_substripe(k+nu, m, w, matrix, 0, erasure_locations, 
                                                   &B_buf[0], &B_buf[k+nu], z, ss_size);
                   */
                    //find if streams is needed
                    planeKernel<<<planeKernelGridSize,planeKernelBlockSize,(k*k + k * m)*sizeof(char)>>>(gf_table, \
                        k, clmsrProfileP->nu, m, clmsrProfileP->w, q, t,\
                        clmsrGpuP->matrix_gpu, decode_matrix_gpu, dm_ids_gpu, erasure_loc_data_gpu, erased_data_size, erasure_loc_coding_gpu, erased_coding_size, planePOnGpuK, pieceSize, pieceSizeMax\
                     );
                   //cudaDeviceSynchronize();
                    //end

                    for(i = 0; i<num_erasures; i++)
                    {
                        x = erasures[i].x;
                        y = erasures[i].y;
                        node_xy = y*q+x;
                        node_sw = y*q+z_vec[y];
                        z_sw = z + ( x - z_vec[y] ) * (int)pow(q,t-1-y);
                        
                        char* B1 = &B_buf[node_xy][z   *sub_chunksize + pieceOffset];
                            
                        //debug
                       /* printf("node_xy: %d, z: %d pieceSizeMax: %d, pieceSize: %d ss_size: %d, pieceOffset %d\n", node_xy, node_sw, pieceSizeMax, pieceSize,ss_size, pieceOffset );
                        printf("sample B1\n");
                        for( int jj = 0; jj < ss_size; jj ++ )
                        {
                            B1[jj] = 10;
                            printf("%u,", B1[jj] );
                        }

                        char *temp = &B_buf[0][0];
                       */ 
                        //char temp1[pieceSize];
                        //char* B2 = &B_buf[node_sw][z_sw*sub_chunksize + pieceOffset];
                        //CUDA_CHECK_RETURN( cudaMemcpyAsync( temp, planeOnGpu[node_xy] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost) );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[node_xy] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost, streams[node_xy]) );
                        //CUDA_CHECK_RETURN( cudaMemcpyAsync( B_buf[0], planeOnGpu[node_xy], 9, cudaMemcpyDeviceToHost) );
                    }
                    //todo new
                    CUDA_CHECK_RETURN(cudaDeviceSynchronize());//todo new: remove this to speed
                    //debug
                    // for(x=0; x < q; x++){
                    //     for(y=0; y<t; y++){
                    //         node_xy = y*q+x;
                    //         if(erased[node_xy] == 1)
                    //         {
                    //             char* B1 = &B_buf[node_xy][z*pieceSize];
                    //             CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[node_xy] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost) );
                    //             printf("\n\nGPU===================================node_xy: %d data( size %d ):\n\n", node_xy,pieceSize );
                    //             printf("\nB1--------------------------------------------------------------\n");
                    //             for( int i = 0; i < ss_size; i ++ )
                    //             {
                    //                 printf("%u,", (unsigned char) B1[i] );
                    //             }
                    //             printf("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n");
                    //         }
                    //     }
                    // }

                }
            }



              //debug
            // cout << "hm_w: " << hm_w << "//////////////////////////////////////////////////////////////////////////\n";
            // cout << "//////////////////////////////////////////////////////////////////////////\n";
            // cout << "//////////////////////////////////////////////////////////////////////////\n";
            // cout << "//////////////////////////////////////////////////////////////////////////\n";
            // cout << "//////////////////////////////////////////////////////////////////////////\n";
            //cout << "get here~~~~~~~~~~~\n";
            //return 0;

        /* Need to get A's from B's*/
            for(z = 0; z< clmsrProfileP->sub_chunk_no; z++)
            {
                if(order[z]==hm_w)
                {
                    //printf(">>>>>>>>>>>>>>in getting A from B %d\n", z);
                    get_plane_vector(q, t, z, z_vec);
                    for(i = 0; i<num_erasures; i++)
                    {
                        x = erasures[i].x;
                        y = erasures[i].y;
                        node_xy = y*q+x;
                        node_sw = y*q+z_vec[y];
                        z_sw = z + ( x - z_vec[y] ) * (int)pow(q,t-1-y);

                        A1 = (node_xy < k+nu) ? &data_ptrs[node_xy][z*ss_size + pieceOffset] : &code_ptrs[node_xy-k-nu][z*ss_size  + pieceOffset];
                        A2 = (node_sw < k+nu) ? &data_ptrs[node_sw][z_sw*ss_size + pieceOffset] : &code_ptrs[node_sw-k-nu][z_sw*ss_size + pieceOffset];


                        char* B1 = &B_buf[node_xy][z   *sub_chunksize + pieceOffset];
                        char* B2 = &B_buf[node_sw][z_sw*sub_chunksize + pieceOffset];

                        if(z_vec[y] != x)
                        { //not a hole-dot pair
                            if(is_erasure_type_1(m, i, erasures, z_vec))
                            {
                                //debughouTab(1,"============A1 from B1 A2\n");
                                CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*1, A2, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );        
                                CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*2, B1, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                                
                                pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],B1A2_A1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                //cudaDeviceSynchronize();       
                                        //debug
                                //CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
                                        
                                CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]], pieceSize, cudaMemcpyDeviceToHost, streams[node_xy]) );
                                //get_type1_A(A1, &B_buf[node_xy][z*ss_size], A2, ss_size);
                            }
                            else
                            {
              // case for type-2 erasure, there is a hole-dot pair in this y column
                                assert(erased[node_sw]==1);
                                //debughouTab(1,"============A1 from B1 B2\n");
                                /*char* B2 = &B_buf[node_sw][z_sw*pieceSize];
                                printf("sample B2::::::----------node_sw: %d, z_sw: %d\n",node_sw, z_sw);
                                for(int i = 0; i < ss_size/2; i ++)
                                {
                                    printf("%u,", B2[i] );
                                } 
*/

                                CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*2, B1, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );        
                                CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*3, B2, pieceSize, cudaMemcpyHostToDevice, streams[node_xy]) );
                                
                                //pieceKernelGamma<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[1]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],GAMMA_INVERSE, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,0,streams[node_xy]>>>( gf_table, clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i], B1B2_A1, pieceSize, pieceSizeMax, clmsrProfileP->w);            
                                //cudaDeviceSynchronize();  
                                            //debug
                                //CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

                                CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]], pieceSize, cudaMemcpyDeviceToHost, streams[node_xy]) );
                                //get_A1_fromB1B2(A1, &B_buf[node_xy][z*ss_size], &B_buf[node_sw][z_sw*ss_size], ss_size);
                            }
                        }
                        else
                        { //for type 0 erasure (hole-dot pair)  copy the B1 to A1
                            //debughouTab(1,"============A1 from B1 red\n");
                            //CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost) );   
                            memcpy(A1, B1, pieceSize);
                        }

                          // printf("\n\nGPU===================================node_xy: %d, erasure_locations[%d]: %d, data( size %d ):\n\n", node_xy, i, erasure_locations[i], pieceSize );
                          // for( int i = 0; i < ss_size; i ++ )
                          // {
                          //   printf("%u,", (unsigned char) A1[i] );
                          // }
                          // printf("\nA2--------------------------------------------------------------\n");
                          // for( int i = 0; i < ss_size; i ++ )
                          // {
                          //   printf("%u,", (unsigned char) A2[i] );
                          // }
                          
                          // char B11[pieceSize],B22[pieceSize];
                          // CUDA_CHECK_RETURN( cudaMemcpyAsync( B11, planeOnGpu[node_xy] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost) );
                          // CUDA_CHECK_RETURN( cudaMemcpyAsync( B22, planeOnGpu[node_xy] + pieceSizeMax*3, pieceSize, cudaMemcpyDeviceToHost) );

                          // printf("\nB1--------------------------------------------------------------\n");
                          // for( int i = 0; i < ss_size; i ++ )
                          // {
                          //   printf("%u,", (unsigned char) B11[i] );
                          // }

                          // printf("\nB2--------------------------------------------------------------\n");
                          // for( int i = 0; i < ss_size; i ++ )
                          // {
                          //   printf("%u,", (unsigned char) B22[i] );
                          // }
                          // printf("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n");

                    }//get A's from B's
                }
            }//plane

            pieceOffset += pieceSize;
        }
    }//hm_w, order

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t2);
    long long deltaT = (t2.tv_sec - t1.tv_sec) * pow(10, 9) + t2.tv_nsec - t1.tv_nsec;

    printf(">>>>>>>>>>>>>>time used: \n%lf s\n\n\n\n", (double)deltaT/pow(10, 9));

    CUDA_CHECK_RETURN(cudaFreeHost(z_vec) );
    CUDA_CHECK_RETURN(cudaFree(decode_matrix_gpu));
    CUDA_CHECK_RETURN(cudaFree(dm_ids_gpu));
    CUDA_CHECK_RETURN(cudaFree(erasure_loc_data_gpu));
    CUDA_CHECK_RETURN(cudaFree(erasure_loc_coding_gpu));
    for( int i = 0; i < qt; i ++ )
    {
        CUDA_CHECK_RETURN(cudaFree(planeOnGpu[i]));
    }

    CUDA_CHECK_RETURN(cudaFree(planePOnGpuK));


    return 0;
}

void SingleGpuRoute::deinit()
{
    for (int i =0; i < clmsrProfileP->q*clmsrProfileP->t; ++i)
    {
        CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
    }

    for (int i =0; i < clmsrProfileP->q*clmsrProfileP->t; ++i)
    {
        CUDA_CHECK_RETURN(cudaEventDestroy(events[i]));
    }

    free(streams);
    free(events);
}

int SingleGpuRoute::init_gf_log_w8_gpu( cudaStream_t streams )
{
/*    int ret = copy_log_to_gpu_w8( gf_table );

    unsigned char log_temp[256];
    unsigned char anti_temp[256*2];
    unsigned char inv_temp[256];

    //debug
    CUDA_CHECK_RETURN( cudaMemcpy(log_temp, gf_table.g_log, 256, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaMemcpy(anti_temp, gf_table.g_anti_log, 256*2, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaMemcpy(inv_temp, gf_table.g_inv, 256, cudaMemcpyDeviceToHost) );

    //cout << " mypt:  " << (int)(&c_log[0]) << endl;
    //printf("your pt0:   %d\n", &c_log[0] );

/*    for( int i = 0; i < 256; i ++ )
    {
        printf("temp---i: %d:\t%d\t %d\n", i, log_temp[i], anti_temp[i] );
    }

    for( int i = 256; i < 256*2; i ++ )
    {
        printf("temp---i: %d:\t%d\n", i, anti_temp[i] );
    }*/


    return copy_log_to_gpu_w8( gf_table );
}

void SingleGpuRoute::compareGf()
{
    init_gf_log_w8_gpu();
    //testGf<<<1,10>>>(gf_table,8);
    testGfHost( 8 );
}
