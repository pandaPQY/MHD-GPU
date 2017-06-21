#include "hip/hip_runtime.h"
#include <stdio.h>
//#include <math.h>
#include "cuda.h"

#include "parameter.h"
#include "array_definition.h" 
#include "cuda_funclist.h"
//extern void checkCUDAError(const char *msg);
//      advectbyzx
__global__ void advectbyzx1_kernel(hipLaunchParm lp, float *adv1_u, float *adv1_b, int *adv1_nx, int *adv1_ny, int *adv1_nz, float *adv1_dt, float *adv1_temp);
__global__ void advectbyzx2_kernel(hipLaunchParm lp, float *adv2_u, float *adv2_b, int *adv2_nx, int *adv2_ny, int *adv2_nz, float *adv2_dt, float *adv2_temp);
__global__ void advectbyzx1b_kernel(hipLaunchParm lp, float *adv1_u, float *adv1_b, int *adv1_nx, int *adv1_ny, int *adv1_nz, float *adv1_dt, float *adv1_temp);
__global__ void advectbyzx2b_kernel(hipLaunchParm lp, float *adv2_u, float *adv2_b, int *adv2_nx, int *adv2_ny, int *adv2_nz, float *adv2_dt, float *adv2_temp);

void cuda_advectbyzx(float *adv_u, float *adv_b, int *adv_nx, int *adv_ny, int *adv_nz, float *adv_dt, float *adv_temp, int *h_adv_nx, int *h_adv_ny, int *h_adv_nz)
{
//      send it to device to calculate
dim3 dimGrid(*h_adv_ny,*h_adv_nz);
dim3 dimBlock(*h_adv_nx);
hipLaunchKernel(HIP_KERNEL_NAME(advectbyzx1_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, adv_u,adv_b,adv_nx,adv_ny,adv_nz,adv_dt,adv_temp);
hipLaunchKernel(HIP_KERNEL_NAME(advectbyzx1b_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, adv_u,adv_b,adv_nx,adv_ny,adv_nz,adv_dt,adv_temp);
hipLaunchKernel(HIP_KERNEL_NAME(advectbyzx2_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, adv_u,adv_b,adv_nx,adv_ny,adv_nz,adv_dt,adv_temp);
hipLaunchKernel(HIP_KERNEL_NAME(advectbyzx2b_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0, adv_u,adv_b,adv_nx,adv_ny,adv_nz,adv_dt,adv_temp);
//
hipDeviceSynchronize();
//
//checkCUDAError("kernel execution in cuda_advectbyzx");
//
}

//      
__global__ void advectbyzx1_kernel(hipLaunchParm lp, float *adv1_u, float *adv1_b, int *adv1_nx, int *adv1_ny, int *adv1_nz, float *adv1_dt, float *adv1_temp)
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
//
__shared__ float adv1_s_u[5*BLOCK_SIZE];
__shared__ float adv1_s_u_jm[5*BLOCK_SIZE];
__shared__ float adv1_s_b[3*BLOCK_SIZE];
//
int adv1_jm;
adv1_jm=(hipBlockIdx_x+(*adv1_ny)-1)%(*adv1_ny);
//
for (int ii=0; ii<5; ii++)
{
	adv1_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=adv1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
for (int ii=0; ii<5; ii++)
{
        adv1_s_u_jm[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=adv1_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,adv1_jm,hipBlockIdx_y)];
}
for (int ii=0; ii<3; ii++)
{
        adv1_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]=adv1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
__syncthreads();
//
float vx;
vx=(adv1_s_u_jm[a2D_FinC(5,hipBlockDim_x,(2-1),hipThreadIdx_x)]+adv1_s_u[a2D_FinC(5,hipBlockDim_x,(2-1),hipThreadIdx_x)])/(adv1_s_u_jm[a2D_FinC(5,hipBlockDim_x,(1-1),hipThreadIdx_x)]+adv1_s_u[a2D_FinC(5,hipBlockDim_x,(1-1),hipThreadIdx_x)]);
//
int adv1_imm,adv1_imp;
adv1_imm=(hipThreadIdx_x+(*adv1_nx)-1)%(*adv1_nx);
adv1_imp=(hipThreadIdx_x+1)%(*adv1_nx);
//
__shared__ float adv1_s_tmp1[BLOCK_SIZE];
adv1_s_tmp1[hipThreadIdx_x]=vx;
__syncthreads();
//
vx=(adv1_s_tmp1[adv1_imm]+adv1_s_tmp1[adv1_imp]+2.0*adv1_s_tmp1[hipThreadIdx_x])/4.0;
//
float b1x;
b1x=adv1_s_b[a2D_FinC(3,hipBlockDim_x,(2-1),hipThreadIdx_x)];
//
//      first tvdb
float vg;
vg=vx;
float b;
b=b1x;
__shared__ float adv1_s_vg[BLOCK_SIZE];
adv1_s_vg[hipThreadIdx_x]=vx;
__syncthreads();
//
float vh;
vh=(adv1_s_vg[hipThreadIdx_x]+adv1_s_vg[adv1_imp])/2.0;
//
__shared__ float adv1_s_tmp2[BLOCK_SIZE];
adv1_s_tmp2[hipThreadIdx_x]=b*vg;
__syncthreads();
float flux1;
if (vh>0) flux1=b*vg;
else flux1=adv1_s_tmp2[adv1_imp];
adv1_s_tmp1[hipThreadIdx_x]=flux1;
__syncthreads();
float b1;
b1=b-(flux1-adv1_s_tmp1[adv1_imm])*(*adv1_dt)/2.0;
//
int ip;
int ipp;
int im;
ip=(hipThreadIdx_x+1)%(*adv1_nx);
ipp=(ip+1)%(*adv1_nx);
im=(hipThreadIdx_x+(*adv1_nx)-1)%(*adv1_nx);
//
float v;
v=vh;
float w;
float wp;
float wm;
__shared__ float adv1_s_b1_tvdb[BLOCK_SIZE];
adv1_s_b1_tvdb[hipThreadIdx_x]=b1;
__syncthreads();
if (v>0)
{
        w=adv1_s_vg[hipThreadIdx_x]*adv1_s_b1_tvdb[hipThreadIdx_x];
        wp=(adv1_s_vg[ip]*adv1_s_b1_tvdb[ip]-w)/2.0;
        wm=(w-adv1_s_vg[im]*adv1_s_b1_tvdb[im])/2.0;
}
else
{
        w=adv1_s_vg[ip]*adv1_s_b1_tvdb[ip];
        wp=(w-adv1_s_vg[ipp]*adv1_s_b1_tvdb[ipp])/2.0;
        wm=(adv1_s_vg[hipThreadIdx_x]*adv1_s_b1_tvdb[hipThreadIdx_x]-w)/2.0;
}
float dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
float flux;
flux=(w+dw)*(*adv1_dt);
//
adv1_s_tmp2[hipThreadIdx_x]=flux;
__syncthreads();
b=b-(flux-adv1_s_tmp2[adv1_imm]);
//      finished tvdb
//
adv1_s_b[a2D_FinC(3,hipBlockDim_x,(2-1),hipThreadIdx_x)]=b;
adv1_s_b[a2D_FinC(3,hipBlockDim_x,(1-1),hipThreadIdx_x)]=adv1_s_b[a2D_FinC(3,hipBlockDim_x,(1-1),hipThreadIdx_x)]-adv1_s_tmp2[adv1_imm];
//
//	send it back to global
for (int ii=0; ii<3; ii++)
{
	adv1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)]=adv1_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)];
}
adv1_temp[a3D_FinC(hipBlockDim_x,hipGridDim_x,hipGridDim_y,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)]=adv1_s_tmp2[adv1_imm];
//
return;
}

__global__ void advectbyzx1b_kernel(hipLaunchParm lp, float *adv1_u, float *adv1_b, int *adv1_nx, int *adv1_ny, int *adv1_nz, float *adv1_dt, float *adv1_temp)
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
int adv1_jm;
adv1_jm=(hipBlockIdx_x+(*adv1_ny)-1)%(*adv1_ny);
//
adv1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,adv1_jm,hipBlockIdx_y)]=adv1_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,adv1_jm,hipBlockIdx_y)]+adv1_temp[a3D_FinC(hipBlockDim_x,hipGridDim_x,hipGridDim_y,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
return;
}

__global__ void advectbyzx2_kernel(hipLaunchParm lp, float *adv2_u, float *adv2_b, int *adv2_nx, int *adv2_ny, int *adv2_nz, float *adv2_dt, float *adv2_temp)
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
//
__shared__ float adv2_s_u[5*BLOCK_SIZE];
__shared__ float adv2_s_u_km[5*BLOCK_SIZE];
__shared__ float adv2_s_b[3*BLOCK_SIZE];
//
int adv2_km;
adv2_km=(hipBlockIdx_y+(*adv2_nz)-1)%(*adv2_nz);
//
for (int ii=0; ii<5; ii++)
{
	adv2_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=adv2_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
for (int ii=0; ii<5; ii++)
{
        adv2_s_u_km[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=adv2_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,adv2_km,hipBlockIdx_y)];
}
for (int ii=0; ii<3; ii++)
{
        adv2_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]=adv2_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
__syncthreads();
//
float vx;
vx=(adv2_s_u_km[a2D_FinC(5,hipBlockDim_x,(2-1),hipThreadIdx_x)]+adv2_s_u[a2D_FinC(5,hipBlockDim_x,(2-1),hipThreadIdx_x)])/(adv2_s_u_km[a2D_FinC(5,hipBlockDim_x,(1-1),hipThreadIdx_x)]+adv2_s_u[a2D_FinC(5,hipBlockDim_x,(1-1),hipThreadIdx_x)]);
//
int adv2_imm,adv2_imp;
adv2_imm=(hipThreadIdx_x+(*adv2_nx)-1)%(*adv2_nx);
adv2_imp=(hipThreadIdx_x+1)%(*adv2_nx);
//
__shared__ float adv2_s_tmp1[BLOCK_SIZE];
adv2_s_tmp1[hipThreadIdx_x]=vx;
__syncthreads();
//
vx=(adv2_s_tmp1[adv2_imm]+adv2_s_tmp1[adv2_imp]+2.0*adv2_s_tmp1[hipThreadIdx_x])/4.0;
//
float b1x;
b1x=adv2_s_b[a2D_FinC(3,hipBlockDim_x,(3-1),hipThreadIdx_x)];
//
//      second tvdb
float vg;
vg=vx;
float b;
b=b1x;
__shared__ float adv2_s_vg[BLOCK_SIZE];
adv2_s_vg[hipThreadIdx_x]=vx;
__syncthreads();
//
float vh;
vh=(adv2_s_vg[hipThreadIdx_x]+adv2_s_vg[adv2_imp])/2.0;
//
__shared__ float adv2_s_tmp2[BLOCK_SIZE];
adv2_s_tmp2[hipThreadIdx_x]=b*vg;
__syncthreads();
float flux1;
if (vh>0) flux1=b*vg;
else flux1=adv2_s_tmp2[adv2_imp];
adv2_s_tmp1[hipThreadIdx_x]=flux1;
__syncthreads();
float b1;
b1=b-(flux1-adv2_s_tmp1[adv2_imm])*(*adv2_dt)/2.0;
//
int ip;
int ipp;
int im;
ip=(hipThreadIdx_x+1)%(*adv2_nx);
ipp=(ip+1)%(*adv2_nx);
im=(hipThreadIdx_x+(*adv2_nx)-1)%(*adv2_nx);
//
float v;
v=vh;
float w;
float wp;
float wm;
__shared__ float adv2_s_b1_tvdb[BLOCK_SIZE];
adv2_s_b1_tvdb[hipThreadIdx_x]=b1;
__syncthreads();
if (v>0)
{
        w=adv2_s_vg[hipThreadIdx_x]*adv2_s_b1_tvdb[hipThreadIdx_x];
        wp=(adv2_s_vg[ip]*adv2_s_b1_tvdb[ip]-w)/2.0;
        wm=(w-adv2_s_vg[im]*adv2_s_b1_tvdb[im])/2.0;
}
else
{
        w=adv2_s_vg[ip]*adv2_s_b1_tvdb[ip];
        wp=(w-adv2_s_vg[ipp]*adv2_s_b1_tvdb[ipp])/2.0;
        wm=(adv2_s_vg[hipThreadIdx_x]*adv2_s_b1_tvdb[hipThreadIdx_x]-w)/2.0;
}
float dw;
dw=0.0;
//
if (wm*wp>0) dw=2.0*wm*wp/(wm+wp);
float flux;
flux=(w+dw)*(*adv2_dt);
//
adv2_s_tmp2[hipThreadIdx_x]=flux;
__syncthreads();
b=b-(flux-adv2_s_tmp2[adv2_imm]);
//      finished tvdb
adv2_s_b[a2D_FinC(3,hipBlockDim_x,(3-1),hipThreadIdx_x)]=b;
adv2_s_b[a2D_FinC(3,hipBlockDim_x,(1-1),hipThreadIdx_x)]=adv2_s_b[a2D_FinC(3,hipBlockDim_x,(1-1),hipThreadIdx_x)]-adv2_s_tmp2[adv2_imm];
for (int ii=0; ii<3; ii++)
{
	adv2_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)]=adv2_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)];
}
adv2_temp[a3D_FinC(hipBlockDim_x,hipGridDim_x,hipGridDim_y,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)]=adv2_s_tmp2[adv2_imm];
//
return;
}

__global__ void advectbyzx2b_kernel(hipLaunchParm lp, float *adv2_u, float *adv2_b, int *adv2_nx, int *adv2_ny, int *adv2_nz, float *adv2_dt, float *adv2_temp)
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
//
int adv2_km;
adv2_km=(hipBlockIdx_y+(*adv2_nz)-1)%(*adv2_nz);
//
adv2_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,adv2_km)]=adv2_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(1-1),hipThreadIdx_x,hipBlockIdx_x,adv2_km)]+adv2_temp[a3D_FinC(hipBlockDim_x,hipGridDim_x,hipGridDim_y,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
//
return;
}
