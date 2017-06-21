#include "hip/hip_runtime.h"
#include <stdio.h>
//#include <math.h>
#include "cuda.h"

#include "parameter.h" 
#include "array_definition.h"
#include "cuda_funclist.h"
//extern void checkCUDAError(const char *msg);
__global__ void transpose13_kernel(hipLaunchParm lp, float *t13_ut, float *t13_bt, float *t13_u, float *t13_b, int *t13_nx, int *t13_ny, int *t13_nz);
__global__ void transpose12_kernel(hipLaunchParm lp, float *t12_ut, float *t12_bt, float *t12_u, float *t12_b, int *t12_nx, int *t12_ny, int *t12_nz);
__global__ void transpose13_nz1_kernel(hipLaunchParm lp, float *t13_nz1_ut, float *t13_nz1_bt, float *t13_nz1_u, float *t13_nz1_b, int *t13_nz1_nx, int *t13_nz1_ny, int *t13_nz1_nz);
__global__ void transpose13_nx1_kernel(hipLaunchParm lp, float *t13_nx1_ut, float *t13_nx1_bt, float *t13_nx1_u, float *t13_nx1_b, int *t13_nx1_nx, int *t13_nx1_ny, int *t13_nx1_nz);

void cuda_transpose13(float *t13_ut, float *t13_bt, float *t13_u, float *t13_b, int *t13_nx, int *t13_ny, int *t13_nz, int *h_t13_nx, int *h_t13_ny, int *h_t13_nz)
{
//      initialization
int Totalthreads = (*h_t13_nx)*(*h_t13_ny)*(*h_t13_nz);
int numThreadsPerBlock = *h_t13_nx;
int numBlocks = Totalthreads/numThreadsPerBlock;
int NumOfU = 5;
int NumOfB = 3;
size_t u_memSize = NumOfU * numBlocks * numThreadsPerBlock * sizeof(float);
size_t b_memSize = NumOfB * numBlocks * numThreadsPerBlock * sizeof(float);
//      send it to device to calculate
dim3 dimGrid(*h_t13_ny,*h_t13_nz);
dim3 dimBlock(*h_t13_nx);
if ((*h_t13_nx)==(*h_t13_nz))
{
        hipLaunchKernel(HIP_KERNEL_NAME(transpose13_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, t13_ut,t13_bt,t13_u,t13_b,t13_nx,t13_ny,t13_nz);
}
else if ((*h_t13_nz)==1)
{
        hipLaunchKernel(HIP_KERNEL_NAME(transpose13_nz1_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, t13_ut,t13_bt,t13_u,t13_b,t13_nx,t13_ny,t13_nz);
}
else if ((*h_t13_nx)==1)
{
        hipLaunchKernel(HIP_KERNEL_NAME(transpose13_nx1_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, t13_ut,t13_bt,t13_u,t13_b,t13_nx,t13_ny,t13_nz);
}
else
{
        printf("nz<>nx not supported\n");
}
//
hipDeviceSynchronize();
//
//checkCUDAError("kernel execution in cuda_transpose13");
//      hipMemcpy
//      from d_ut to d_u, in device
//      from d_bt to d_b, in device
hipMemcpy(t13_u,t13_ut, u_memSize, hipMemcpyDeviceToDevice );
hipMemcpy(t13_b,t13_bt, b_memSize, hipMemcpyDeviceToDevice );
//
//checkCUDAError("memcpy: from device to device, in cuda_transpose13");
//
}

__global__ void transpose13_kernel(hipLaunchParm lp, float *t13_ut, float *t13_bt, float *t13_u, float *t13_b, int *t13_nx, int *t13_ny, int *t13_nz)
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
t13_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(1-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(2-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(4-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(3-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(4-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(5-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(5-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
t13_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(1-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(2-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(3-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
return;
}

__global__ void transpose13_nz1_kernel(hipLaunchParm lp, float *t13_nz1_ut, float *t13_nz1_bt, float *t13_nz1_u, float *t13_nz1_b, int *t13_nz1_nx, int *t13_nz1_ny, int *t13_nz1_nz)
{
/*
i = hipThreadIdx_x
j = hipBlockIdx_x
k = hipBlockIdx_y
nx = hipBlockDim_x
ny = hipGridDim_x
nz = hipGridDim_y
*/
//	transpose12
t13_nz1_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(1-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(2-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(3-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(4-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(4-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_ut[a4D_FinC(5,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(5-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(5-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
t13_nz1_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(1-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(2-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
t13_nz1_bt[a4D_FinC(3,hipGridDim_x,hipBlockDim_x,hipGridDim_y,(3-1),hipBlockIdx_x,hipThreadIdx_x,hipBlockIdx_y)]=t13_nz1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
//	second part
float temp1,temp2,temp3;
temp1=t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(2-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
temp2=t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(3-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
temp3=t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(4-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(2-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)]=temp3;
t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(3-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)]=temp1;
t13_nz1_ut[a4D_FinC(5,1,hipGridDim_x,hipBlockDim_x,(4-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)]=temp2;
//
temp1=t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(1-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
temp2=t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(2-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
temp3=t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(3-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)];
t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(1-1),(1-1),hipBlockIdx_x,hipThreadIdx_x)]=temp3;
t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(1-1),(2-1),hipBlockIdx_x,hipThreadIdx_x)]=temp1;
t13_nz1_bt[a4D_FinC(3,1,hipGridDim_x,hipBlockDim_x,(1-1),(3-1),hipBlockIdx_x,hipThreadIdx_x)]=temp2;
//
return;
}

__global__ void transpose13_nx1_kernel(hipLaunchParm lp, float *t13_nx1_ut, float *t13_nx1_bt, float *t13_nx1_u, float *t13_nx1_b, int *t13_nx1_nx, int *t13_nx1_ny, int *t13_nx1_nz)
{
/*
i = hipThreadIdx_x
j = hipBlockIdx_x
k = hipBlockIdx_y
nx = hipBlockDim_x
ny = hipGridDim_x
nz = hipGridDim_y
*/
//      transpose12
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(1-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_u[a4D_FinC(5,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(1-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(2-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_u[a4D_FinC(5,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(3-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(3-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_u[a4D_FinC(5,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(2-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(4-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_u[a4D_FinC(5,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(4-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(5-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_u[a4D_FinC(5,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(5-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
//
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(1-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_b[a4D_FinC(3,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(2-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(2-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_b[a4D_FinC(3,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(1-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,hipBlockDim_x,(3-1),hipBlockIdx_y,hipBlockIdx_x,hipThreadIdx_x)]=t13_nx1_b[a4D_FinC(3,hipGridDim_x,hipGridDim_y,hipBlockDim_x,(3-1),hipBlockIdx_x,hipBlockIdx_y,hipThreadIdx_x)];
//
//	second part
float temp1,temp2,temp3;
temp1=t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(2-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
temp2=t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(3-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
temp3=t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(4-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(2-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp3;
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(3-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp1;
t13_nx1_ut[a4D_FinC(5,hipGridDim_y,hipGridDim_x,1,(4-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp2;
//
temp1=t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(1-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
temp2=t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(2-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
temp3=t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(3-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))];
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(1-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp3;
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(2-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp1;
t13_nx1_bt[a4D_FinC(3,hipGridDim_y,hipGridDim_x,1,(3-1),hipBlockIdx_y,hipBlockIdx_x,(1-1))]=temp2;
//
return;
}
