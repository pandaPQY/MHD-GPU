#include "hip/hip_runtime.h"
#include <stdio.h>
//#include <math.h>
#include "cuda.h"

#include "parameter.h" 
#include "array_definition.h"
#include "cuda_funclist.h"
//extern void checkCUDAError(const char *msg);
__global__ void transpose12_kernel(hipLaunchParm lp, float *t12_ut, float *t12_bt, float *t12_u, float *t12_b, int *t12_nx, int *t12_ny, int *t12_nz);
//	

void cuda_transpose12(float *t12_ut, float *t12_bt, float *t12_u, float *t12_b, int *t12_nx, int *t12_ny, int *t12_nz, int *h_t12_nx, int *h_t12_ny, int *h_t12_nz)
{
//      initialization
int Totalthreads = (*h_t12_nx)*(*h_t12_ny)*(*h_t12_nz);
int numThreadsPerBlock = *h_t12_nx;
int numBlocks = Totalthreads/numThreadsPerBlock;
int NumOfU = 5;
int NumOfB = 3;
size_t u_memSize = NumOfU * numBlocks * numThreadsPerBlock * sizeof(float);
size_t b_memSize = NumOfB * numBlocks * numThreadsPerBlock * sizeof(float);
//      send it to device to calculate
dim3 dimGrid(*h_t12_ny,*h_t12_nz);
dim3 dimBlock(*h_t12_nx);
hipLaunchKernel(HIP_KERNEL_NAME(transpose12_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, t12_ut,t12_bt,t12_u,t12_b,t12_nx,t12_ny,t12_nz);
//
hipDeviceSynchronize();
//
//checkCUDAError("kernel execution in cuda_transpose12");
//	hipMemcpy
//	from d_ut to d_u, in device
//	from d_bt to d_b, in device
hipMemcpy(t12_u,t12_ut, u_memSize, hipMemcpyDeviceToDevice );
hipMemcpy(t12_b,t12_bt, b_memSize, hipMemcpyDeviceToDevice );
//
//checkCUDAError("memcpy: from device to device, in cuda_transpose12");
//
}

__global__ void transpose12_kernel(hipLaunchParm lp, float *t12_ut, float *t12_bt, float *t12_u, float *t12_b, int *t12_nx, int *t12_ny, int *t12_nz)
{
/*
two dimensional array of blocks on grid where each block has one dimensional array of threads:
UniqueBlockIndex = hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x;
UniqueThreadIndex = UniqueBlockIndex * hipBlockDim_x + hipThreadIdx_x;
*/
/*
i = hipThreadIdx_x
j = hipBlockIdx_x
k = hipBlockIdx_y
nx = hipBlockDim_x
ny = hipGridDim_x
nz = hipGridDim_y
*/
t12_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(1-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(2-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(3-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(4-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(4-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(5-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(5-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
t12_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(1-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(2-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t12_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(3-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t12_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
return;
}

//	
//-----------------------
/*
   Fortran subroutine arguments are passed by references.
   call fun( array_a, array_b, N) will be mapped to
   function (*a, *b, *N);
*/
extern "C" void cuda_transpose12_(float *h_ut, float *h_bt, float *h_u, float *h_b, int *h_nx, int *h_ny, int *h_nz, float *h_dt)
{
int Totalthreads = (*h_nx)*(*h_ny)*(*h_nz);
int numThreadsPerBlock = *h_nx;
int numBlocks = Totalthreads/numThreadsPerBlock;
int NumOfU = 5;
int NumOfB = 3;
//      intialize
size_t u_memSize = NumOfU * numBlocks * numThreadsPerBlock * sizeof(float);
size_t b_memSize = NumOfB * numBlocks * numThreadsPerBlock * sizeof(float);
//
float *d_u, *d_b;
hipMalloc( (void **) &d_u, u_memSize );
hipMalloc( (void **) &d_b, b_memSize );
hipMemcpy( d_u, h_u, u_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_b, h_b, b_memSize, hipMemcpyHostToDevice );
//
float *d_ut, *d_bt;
hipMalloc( (void **) &d_ut, u_memSize );
hipMalloc( (void **) &d_bt, b_memSize );
//
int *d_nx,*d_ny,*d_nz;
size_t n_memSize = sizeof(int);
hipMalloc( (void **) &d_nx, n_memSize );
hipMalloc( (void **) &d_ny, n_memSize );
hipMalloc( (void **) &d_nz, n_memSize );
hipMemcpy( d_nx, h_nx, n_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_ny, h_ny, n_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_nz, h_nz, n_memSize, hipMemcpyHostToDevice );
//
dim3 dimGrid(*h_ny,*h_nz);
dim3 dimBlock(numThreadsPerBlock);
hipLaunchKernel(HIP_KERNEL_NAME(transpose12_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0,  d_ut, d_bt, d_u, d_b, d_nx, d_ny, d_nz);
//
hipDeviceSynchronize();
//
//checkCUDAError("kernel execution");
//
//	find the max
hipMemcpy( h_ut, d_ut, u_memSize, hipMemcpyDeviceToHost );
hipMemcpy( h_bt, d_bt, b_memSize, hipMemcpyDeviceToHost );
//
//checkCUDAError("memcpy");
//
hipFree(d_u);
hipFree(d_b);
hipFree(d_nx);
hipFree(d_ny);
hipFree(d_nz);
hipFree(d_ut);
hipFree(d_bt);
//
return;
}



