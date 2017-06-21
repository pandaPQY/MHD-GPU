#ifndef CUDA_FUNCTION_H
#define CUDA_FUNCTION_H 
#include "/opt/rocm/hip/include/hip/hcc_detail/cuda/cuda.h"
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

#endif

