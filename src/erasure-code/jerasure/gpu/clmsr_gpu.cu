#include "clmsr_gpu.h"

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "jerasure_base.h"
#include "gf_base.h"

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
#define GAMMA 5
#define GAMMA_INVERSE 6

#define talloc(type, num) (type *) malloc(sizeof(type)*(num))

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

        //done: init B_Buf and pin it
        CUDA_CHECK_RETURN(cudaHostAlloc(&B_buf, clmsrProfile.q*clmsrProfile.t*sizeof(char*), cudaHostAllocPortable));

        assert(B_buf);

        for(int i = 0; i < clmsrProfile.q*clmsrProfile.t; i++)
        {
            //checkCudaErrors(cudaHostAlloc(&h_data_in[i], memsize, cudaHostAllocPortable));
            if(B_buf[i]==NULL)
            {
                CUDA_CHECK_RETURN(cudaHostAlloc(&B_buf[i], (size_t)(clmsrProfile.subChunkSize*clmsrProfile.sub_chunk_no), cudaHostAllocPortable));
                assert(B_buf[i]);
            }
        }


        //done: pin memory
        pinMemory(repaired_data, clmsrProfile.chunkSize );
        pinMemory(helper_data, clmsrProfile.subChunkSize*repair_sub_chunks_ind.size() );

        //done: get GpuInfo

        //todo try to put it in the shared memory;
        CUDA_CHECK_RETURN(cudaMalloc(&matrix_gpu, sizeof(int)*clmsrProfile.k*clmsrProfile.m));
        CUDA_CHECK_RETURN(cudaMemcpy(matrix_gpu, clmsrProfile.matrix, sizeof(int)*(clmsrProfile.k*clmsrProfile.m), cudaMemcpyHostToDevice));

     }

ClmsrGpu::~ClmsrGpu()
{
    //todo: free B_buf and unpin it
    for(int i = 0; i < clmsrProfile.q*clmsrProfile.t; i++)
    {
        //checkCudaErrors(cudaHostAlloc(&h_data_in[i], memsize, cudaHostAllocPortable));
        cudaFreeHost(B_buf[i]);
    }

    cudaFreeHost(B_buf);

    //done: unpinmemory
    unpinMemory(repaired_data );
    unpinMemory(helper_data );


    cudaFree(matrix_gpu);
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

    init();

}

SingleGpuRoute::~SingleGpuRoute()
{
    deinit();
}


int SingleGpuRoute::__getPieceSize( int i )
{
    //todo: think about thread
    int baseSize = subSubChunkSize%pieceCount == 0;

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

    for( int i = 0; i < STREAM_NUM; i ++ )
    {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
    }

    for( int i = 0; i < EVENT_NUM; i ++ )
    {
        CUDA_CHECK_RETURN(cudaEventCreate(&events[i]));
    }
    //cudaEventRecord(events[i], streams[i]);
    //todo: init stream and events
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

__global__ void pieceKernelGamma( char gamma, char** dataPt,  int nodeId, int calType, int dataSize, int patchSize, int w )
{
    load_tables( threadIdx, blockDim );

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    int tmatrix[4];


    char* A[2];
    char* dest[2];

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

            int gamma_square = galois_single_multiply_gpu_logtable_w8(gamma, gamma);
            int gamma_det_inv = galois_single_divide_gpu_logtable_w8(1, 1 ^ (gamma_square));
            tmatrix[0] = gamma_det_inv;
            tmatrix[1] = galois_single_multiply_gpu_logtable_w8(gamma,gamma_det_inv);
            tmatrix[2] = tmatrix[1];
            tmatrix[3] = tmatrix[0];



            A[0] = dataPt[nodeId] + patchSize*2; //B2
            A[1] = dataPt[nodeId] + patchSize*3; //B1

            dest[0] = dataPt[nodeId] + patchSize*0;
            dest[1] = dataPt[nodeId] + patchSize*1;

            break;

        default:
            printf("error Case\n");
            break;
    }


    //jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, A, dest, size);
    //jerasure_matrix_dotprod(2, w, &tmatrix[2], NULL, 3, A, dest, size);
    for( int i = idx; i < dataSize; i += threadsNum )
    {
        //todo: optmize a*1 case;
        dest[0][i] = galois_single_multiply_gpu_logtable_w8(A[0][i], tmatrix[0]) ^ galois_single_multiply_gpu_logtable_w8(A[1][i], tmatrix[1]);
        dest[1][i] = galois_single_multiply_gpu_logtable_w8(A[0][i], tmatrix[2]) ^ galois_single_multiply_gpu_logtable_w8(A[1][i], tmatrix[3]);
    }
}

__global__ void pieceKernel( char gamma, char** dataPt,  int nodeId, int calType, int dataSize, int patchSize, int w )
{
    load_tables( threadIdx, blockDim );

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    int tmatrix[2];
    char* in_dot[2];
    char* dest[1];
    //todo: make the preceed all the same by changing the pointer and matrix;
    //for( int i = idx; i < dataSize; i += threadsNum )
    //todo: if use this patern, you must make sure that blockDim.x = blockSize;
/*    for( int i = blockStart + threadIdx.x; i < dataSize && i < blockStart + dataSizeBlock; i += blockDim.x )
    {

    }*/
    switch(calType)
    {
        case A1A2_B1        :

            tmatrix[0] = 1;
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId];  //A1
            in_dot[1] = dataPt[nodeId] + patchSize; //A2

            dest[0] = dataPt[nodeId] + patchSize*2; //B1
            break;

        case A1B2_B1        :

            int gamma_square = galois_single_multiply_gpu_logtable_w8(gamma, gamma);

            tmatrix[0] = (1 ^ gamma_square);
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId];  //A1
            in_dot[1] = dataPt[nodeId] + patchSize*3; //B2

            dest[0] = dataPt[nodeId] + patchSize*2; //B1
            break;

        case B1A2_A1        :

            tmatrix[0] = 1;
            tmatrix[1] = gamma;

            in_dot[0] = dataPt[nodeId] + patchSize*2;  //B1
            in_dot[1] = dataPt[nodeId] + patchSize*1; //A2

            dest[0] = dataPt[nodeId]; //A1
            break;

        case A1B1_A2        :

            tmatrix[0] = galois_single_divide_gpu_logtable_w8(1,gamma);
            tmatrix[1] = tmatrix[0];

            in_dot[0] = dataPt[nodeId] + patchSize*2;  //B1
            in_dot[1] = dataPt[nodeId] + patchSize*1; //A2

            dest[0] = dataPt[nodeId]; //A1

        default:
            printf("error Case\n");
            break;
    }
    
    //jerasure_matrix_dotprod(2, w, &tmatrix[0], NULL, 2, in_dot, dest, size);
    //dest[i] = in_dot[0][i] * tmatrix[0] + in_dot[1][i] * tmatrix[1]; 
    for( int i = idx; i < dataSize; i += threadsNum )
    {
        //todo: optmize a*1 case;
        dest[i] = galois_single_multiply_gpu_logtable_w8(in_dot[0][i], tmatrix[0]) ^ galois_single_multiply_gpu_logtable_w8(in_dot[1][i], tmatrix[1]);    
    }
}



/*planeKernel<<<planeKernelGridSize,planeKernelBlockSize,streams[1]>>>(\
                        clmsrProfileP->k, clmsrProfileP->nu, clmsrProfileP->m, clmsrProfileP->w,\
                        clmsrGpuP->matrix_gpu, erasure_locations_gpu, planePOnGpuK, pieceSize
                     );*/

__global__ void planeKernel( const int k, int nu, const int m, int w, int q, int t, int* matrix, int* decode_matrix, \
    int* dm_ids, int* erasure_loc_data, int erased_data_size, int* erasure_loc_coding, int erased_coding_size, char** dataPt, int dataSize, int patchSize )
{
    load_tables( threadIdx, blockDim );

    unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned threadsNum = gridDim.x*blockDim.x;

    //load matrix to gpu shared memery;
    __shared__ char sh_decode_matrix[(const int)(k*k)];
    __shared__ char sh_matrix[(const int)(k*m)];


    for( int t = idx; t < k*k; t += threadsNum )
    {
        sh_decode_matrix[t] = (char) decode_matrix[t];
    }

    for( int t = idx; t < k*m; t += threadsNum )
    {
        sh_matrix[t] = (char) matrix[t];
    }
    
    __syncthreads();


/*  for (i = 0; i < k; i++) {
    if (erased[i]) {
      jerasure_matrix_dotprod_substripe(k, w, decoding_matrix+(i*k), dm_ids, i, data_ptrs, coding_ptrs, z, ss_size);
      edd--;
    }
  }

   Finally, re-encode any erased coding devices 

  for (i = 0; i < m; i++) {
    if (erased[k+i]) {
      jerasure_matrix_dotprod_substripe(k, w, matrix+(i*k), NULL, i+k, data_ptrs, coding_ptrs, z, ss_size);
    }
  }*/

    int * data_now;
    int * erasure_now;

    for( int i = 0 ; i < k; i ++ )
    {
        data_now = dataPt[dm_ids[i]] + patchSize*2;//B1 is what we want do RS

        for( int j = 0; j < erased_data_size; j ++ )
        {
            erasure_now = dataPt[erasure_loc_data[j]] + patchSize*2;
            
            for( int t = idx; t < dataSize; t += threadsNum )
            {
                //todo: ensure that dest[i] is all clean first time;
                if( i == 0 )
                {
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], (char)(sh_decode_matrix + j*k + i));
                }
                else
                {
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], (char)(sh_decode_matrix + j*k + i));
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
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], (char)(sh_matrix + j*k + i));
                }
                else
                {
                    erasure_now[t] = galois_single_multiply_gpu_logtable_w8( data_now[t], (char)(sh_matrix + j*k + i));
                }    
            }
        }
    }
}

int SingleGpuRoute::doRepair()
{
    FT(SingleGpuRoute::doRepair);
    const int qt = clmsrProfileP->q * clmsrProfileP->t;
    int* z_vec[clmsrProfileP->t];

    CUDA_CHECK_RETURN( cudaHostAlloc(&z_vec, clmsrProfileP->t * sizeof(int), cudaHostAllocPorted) );

    map<int, set<int> > ordered_planes;
    map<int, int> repair_plane_to_ind;
    int order = 0;
    int x,y, node_xy, node_sw, z_sw;
    char *A1, *A2, *B1, *B2;
    int count_retrieved_sub_chunks = 0;
    int num_erased = 0;
    int subChunkSize = clmsrProfileP->subChunkSize;

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
    
    bool init_matrix;

    dm_ids = talloc(int, clmsrProfileP->k);
    decode_matrix = talloc(int, clmsrProfileP->k*clmsrProfileP->k);
    erased = talloc(int, clmsrProfileP->k + clmsrProfileP->m);
    erasure_loc_data = talloc(int, qt);
    erasure_loc_coding = talloc(int, qt);

    CUDA_CHECK_RETURN(cudaMalloc(&decode_matrix_gpu, clmsrProfileP->k * clmsrProfileP->k *sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&dm_ids_gpu, clmsrProfileP->k * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_data_gpu, qt));
    CUDA_CHECK_RETURN(cudaMalloc(&erasure_loc_coding_gpu, qt));

    //malloc the data temp space on the gpu a piece has 4 things: A1,A2,B1,B2;
    
    int pieceSizeMax = __getPieceSize(0);
    char* planeOnGpu[qt];
    for( int i = 0; i < qt; i ++ )
    {
        CUDA_CHECK_RETURN(cudaMalloc(&planeOnGpu[i], pieceSizeMax*4));
    }

    char** planePOnGpuK;

    CUDA_CHECK_RETURN(cudaMalloc(&planePOnGpuK, qt*sizeof(char*)));

    CUDA_CHECK_RETURN(cudaMemcpyAsync(planePOnGpuK, planeOnGpu, qt*sizeof(char*), cudaMemcpyHostToDevice,streams[0]));

    int plane_count = 0;
    int erasure_locations[qt];

    //get order of all planes
    for(map<int,int>::iterator i = clmsrGpuP->repair_sub_chunks_ind.begin(); i != clmsrGpuP->repair_sub_chunks_ind.end(); ++i)
    {
        get_plane_vector(q,t, i->second, z_vec);
        order = 0;
        //check across all erasures
        for(map<int,char*>::iterator j = repaired_data.begin(); j != repaired_data.end(); ++j)
        {
          if(j->first%q == z_vec[j->first/q])order++;
        }
        assert(order>0);
        ordered_planes[order].insert(i->second);
        //repair_plane_to_ind[i->second] = i->first;
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

                int pieceOffset = 0;
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

                            node_xy = y*q + x;
                            z_sw = (*z) + (x - z_vec[y])*pow_int(q,t-1-y);
                            node_sw = y*q + z_vec[y];

                            if( (repaired_data.find(node_xy) != repaired_data.end()) )
                            {//case of erasure, aloof node can't get a B.

                                erasure_locations[num_erased] = node_xy;
                                num_erased++;
                                if( repaired_data.find(node_sw) != repaired_data.end() )
                                {//node_sw must be a helper
                                    if(x > z_vec[y])//todo: check if is right
                                    {
                                        B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*3,  B2, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                    }
                                }
                                else
                                {
                                    assert( helper_data.find(node_sw) != helper_data.end() );
                                    A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];

                                    if( z_vec[y] != x)
                                    {
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*1, A2, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
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

                                
                                //dout(10) << "current node=" << node_xy << " plane="<< *z << " node_sw=" << node_sw << " plane_sw="<< z_sw << dendl;
                                //consider this as an erasure, if A2 not found.
                                //todo: 这里的判断条件不太对啊， 没有判断你是一个helper 但你要的A2在erasred里的情况
                                if(repair_plane_to_ind.find(z_sw) == repair_plane_to_ind.end())
                                {
                                    erasure_locations[num_erased] = node_xy;
                                    //dout(10)<< num_erased<< "'th erasure of node " << node_xy << " = (" << x << "," << y << ")" << dendl;
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
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                A1, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax, A2, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                        
                                        //todo: find a parameter
                                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,node_xy, A1A2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);


                                        //get_B1_fromA1A2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, A2,sub_chunksize);
                                    }
                                    else if(aloof_nodes.find(node_sw) != aloof_nodes.end())
                                    {
                                        B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];

                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                  A1, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*3, B2, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                        
                                        //todo: find a parameter
                                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,node_xy,A1B2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                        //get_B1_fromA1B2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, B2, sub_chunksize);
                                    }
                                    else
                                    {
                                        assert(helper_data.find(node_sw) != helper_data.end());
                      //dout(10) << "obtaining B1 from A1 A2 for node: " << node_xy << " on plane:" << *z << dendl;
                                        A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                                        if( z_vec[y] != x)
                                        {
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                  A1, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*1, A2, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                        
                                            //todo: find a parameter
                                            pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,node_xy,A1A2_B1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                            //get_B1_fromA1A2(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, A2, sub_chunksize);
                                        }
                                        else
                                        {
                                            //CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy],                  A1, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync(planeOnGpu[node_xy] + pieceSizeMax*2, A1, pieceSize, cudaMemcpyHostToDevice, streams[0]) );
                                            //pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>(planePOnGpuK,node_xy,A1A2);
                                            //pieceKernel<<<>>>(A1A2);red point
                                            //memcpy(&B_buf[node_xy][repair_plane_to_ind[*z]*sub_chunksize], A1, sub_chunksize);
                                        }
                                    }
                                }

                            }


                        }//y
                    }//x

                    erasure_locations[num_erased] = -1;
                    //int erasuresxy[num_erased];
                    //get_erasure_coordinates(erasure_locations, erasuresxy, num_erased);
                    //we obtained all the needed B's
                    assert(num_erased <= m);
                    


                    if(!init_matrix)
                    {
                        init_matrix = true;
                        
                        erasures_to_erased (clmsrProfileP->k, clmsrProfileP->m, erasure_locations, erased );

                        if (jerasure_make_decoding_matrix(clmsrProfileP->k, clmsrProfileP->m, clmsrProfileP->w, clmsrProfileP->matrix, erased, decode_matrix, dm_ids) < 0) 
                        {
                          printf("Can not get decoding matrix!!!\n");
                          return -1;
                        }
                        
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(decode_matrix_gpu, decode_matrix, clmsrProfileP->k*clmsrProfileP->k*sizeof(int), cudaMemcpyHostToDevice, streams[0]) );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(dm_ids_gpu, dm_ids, clmsrProfileP->k*sizeof(int), cudaMemcpyHostToDevice, streams[0]) );
                        

                        erased_data_size = full_erased_list_data( clmsrProfileP->k, clmsrProfileP->m, erasure_loc_data, erased );
                        erased_coding_size = full_erased_list_data( clmsrProfileP->k, clmsrProfileP->m, erasure_loc_coding, erased );

                        CUDA_CHECK_RETURN( cudaMemcpyAsync(erasure_loc_data_gpu, erasure_loc_data, qt*sizeof(int), cudaMemcpyHostToDevice, streams[0]) );
                        CUDA_CHECK_RETURN( cudaMemcpyAsync(erasure_loc_coding_gpu, erasure_loc_coding, qt*sizeof(int), cudaMemcpyHostToDevice, streams[0]) );
                   
                    }

                    //dout(10) << "going to decode for B's in repair plane "<< *z << " at index " << repair_plane_to_ind[*z] << dendl;
                    //jerasure_matrix_decode_substripe(k+nu, m, w, matrix, 0, erasure_locations, &B_buf[0], &B_buf[k+nu], repair_plane_to_ind[*z], sub_chunksize);
                                            
                    planeKernel<<<planeKernelGridSize,planeKernelBlockSize,streams[1]>>>(\
                        clmsrProfileP->k, clmsrProfileP->nu, clmsrProfileP->m, clmsrProfileP->w, clmsrProfileP->q, clmsrProfileP->t,\
                        clmsrGpuP->matrix_gpu, decode_matrix_gpu, dm_ids_gpu, erasure_loc_data_gpu, erased_data_size, erasure_loc_coding_gpu, erased_coding_size, planePOnGpuK, pieceSize, pieceSizeMax\
                     );//decode;

                    for(int i = 0; i < num_erased; i++)
                    {
                        x = erasure_locations[i]%q;
                        y = erasure_locations[i]/q;
                        //dout(10) << "B symbol recovered at (x,y) = (" << x <<","<<y<<")"<<dendl;
                        //dout(10) << "erasure location " << erasure_locations[i] << dendl;
                        node_sw = y*q+z_vec[y];
                        z_sw = (*z) + (x - z_vec[y]) * pow_int(q,t-1-y);


                        

                        //make sure it is not an aloof node before you retrieve repaired_data
                        if( aloof_nodes.find(erasure_locations[i]) == aloof_nodes.end())
                        {
                            if(x == z_vec[y] )
                            {//hole-dot pair (type 0)
                            //dout(10) << "recovering the hole dot pair/lost node in repair plane" << dendl;
                                A1 = &repaired_data[erasure_locations[i]][*z*sub_chunksize + pieceOffset];
                                CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                                //memcpy(A1, B1, sub_chunksize);
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
                                        if(x < z_vec[y])//todo: check this is ensure!!!!
                                        {//recover both A1 and A2 here
                                            A2 = &repaired_data[node_sw][z_sw*sub_chunksize + pieceOffset];
                                            //B2 = &B_buf[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];
                                            

                                            pieceKernelGamma<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],GAMMA_INVERSE, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]],                  pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync( A2, planeOnGpu[erasure_locations[i]] + pieceSizeMax*1, pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                                
                                            
                                            //gamma_inverse_transform(A1, A2, B1, B2, sub_chunksize);
                                            count_retrieved_sub_chunks = count_retrieved_sub_chunks + 2;
                                        }
                                        else
                                        {
                                            B1 = &B_buf[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize + pieceOffset];
                                            CUDA_CHECK_RETURN( cudaMemcpyAsync( B1, planeOnGpu[erasure_locations[i]] + pieceSizeMax*2, pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                                        }
                                    }
                                    else//node_sw是一个helper节点：　这里不可能是一个aloof节点，因为同一个y-section的都是helper节点．
                                    {
                                        //dout(10) << "repaired_data" << repaired_data << dendl;
                                        //A2 for this particular node is available
                                        assert(helper_data.find(node_sw) != helper_data.end());
                                        assert(repair_plane_to_ind.find(z_sw) !=  repair_plane_to_ind.end());
                                        //A2 = &helper_data[node_sw][repair_plane_to_ind[z_sw]*sub_chunksize + pieceOffset];

                                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],B1A2_A1, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync( A1, planeOnGpu[erasure_locations[i]], pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                            
                                        //get_type1_A(A1, B1, A2, sub_chunksize);
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
                                        //i got to recover A2, if z_sw was already there
                                        //dout(10) << "recovering A2 of node:" << node_sw << " at location " << z_sw << dendl;
                                        pieceKernel<<<pieceKernelGridSize,pieceKernelBlockSize,streams[1]>>>( clmsrProfileP->gamma,planePOnGpuK,erasure_locations[i],A1B1_A2, pieceSize, pieceSizeMax, clmsrProfileP->w);
                                        CUDA_CHECK_RETURN( cudaMemcpyAsync( A2, planeOnGpu[erasure_locations[i]] + pieceSizeMax*1, pieceSize, cudaMemcpyDeviceToHost, streams[2]) );
                            

                                        //A1 = &helper_data[erasure_locations[i]][repair_plane_to_ind[*z]*sub_chunksize];

                                        //get_type2_A(A2, B1, A1, sub_chunksize);
                                        count_retrieved_sub_chunks++;
                                    }
                                }
                            }//type-1 erasure recovered.
                        }//not an aloof node
                    }//erasures

           //dout(10) << "repaired data after decoding at plane: " << *z << " "<< repaired_data << dendl;
           //dout(10) << "helper data after decoding at plane: " << *z << " "<< helper_data << dendl;
                    pieceOffset += pieceSize;
                }
            }//planes of a particular order

        }
    }
    assert(repair_sub_chunks_ind.size() == (unsigned)plane_count);
    assert(sub_chunk_no*repaired_data.size() == (unsigned)count_retrieved_sub_chunks);

    //dout(10) << "repaired_data = " << repaired_data << dendl;

    //todo : something route like this
    //todo : ensure that size is big enough.

    /*
    for layers
    {
        for node
        {
            for pieces
            {
                tranfer node;
                cal node'b
                add result to RS
                transfer necessary b back;
            }
        }
        cal the A of origin Miss part
        transfer it back
    }
    */
}


void SingleGpuRoute::doDecode()
{

}

void SingleGpuRoute::deinit()
{
    for (int i =0; i < STREAM_NUM; ++i)
    {
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream[i]));
    }

    for (int i =0; i < EVENT_NUM; ++i)
    {
        CUDA_CHECK_RETURN(cudaEventDestroy(events[i]));
    }
}
