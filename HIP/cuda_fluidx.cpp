#include "hip/hip_runtime.h"
#include <stdio.h>
//#include <math.h>
#include "cuda.h"

#include "parameter.h" 
#include "array_definition.h"
#include "cuda_funclist.h"
//extern void checkCUDAError(const char *msg);
__global__ void fluidx_kernel(hipLaunchParm lp, float *flu_u, float *flu_b, int *flu_nx, int *flu_ny, int *flu_nz, float *flu_dt);
__device__ void mhdflux(float *mf_v, float *mf_c, float *mf_u, float *mf_b, int *mf_n);

void cuda_fluidx(float *fluidx_u, float *fluidx_b, int *fluidx_nx, int *fluidx_ny, int *fluidx_nz, float *fluidx_dt, int *h_fluidx_nx, int *h_fluidx_ny, int *h_fluidx_nz)
{
//      send it to device to calculate
dim3 dimGrid(*h_fluidx_ny,*h_fluidx_nz);
dim3 dimBlock(*h_fluidx_nx);
hipLaunchKernel(HIP_KERNEL_NAME(fluidx_kernel), dim3(dimGrid), dim3(dimBlock ), 0, 0,  fluidx_u, fluidx_b, fluidx_nx, fluidx_ny, fluidx_nz, fluidx_dt);
//
hipDeviceSynchronize();
//
//checkCUDAError("kernel execution in cuda_fluidx");
//
}

__global__ void fluidx_kernel(hipLaunchParm lp, float *flu_u, float *flu_b, int *flu_nx, int *flu_ny, int *flu_nz, float *flu_dt)
{
/*
two dimensional array of blocks on grid where each block has one dimensional array of threads:
UniqueBlockIndex = hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x;
UniqueThreadIndex = UniqueBlockIndex * hipBlockDim_x + hipThreadIdx_x;
*/
//
__shared__ float flu_s_b[3*BLOCK_SIZE];
__shared__ float flu_s_u[5*BLOCK_SIZE];
__shared__ float flu_s_jp_b2[BLOCK_SIZE];
__shared__ float flu_s_kp_b3[BLOCK_SIZE];
//
for (int ii=0; ii<3; ii++)
{
	flu_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]=flu_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
//
for (int ii=0; ii<5; ii++)
{
	flu_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=flu_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)];
}
//
int flu_jp,flu_kp;
flu_jp=(hipBlockIdx_x+1)%(*flu_ny);
flu_kp=(hipBlockIdx_y+1)%(*flu_nz);
flu_s_jp_b2[hipThreadIdx_x]=flu_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(2-1),hipThreadIdx_x,flu_jp,hipBlockIdx_y)];
flu_s_kp_b3[hipThreadIdx_x]=flu_b[a4D_FinC(3,hipBlockDim_x,hipGridDim_x,hipGridDim_y,(3-1),hipThreadIdx_x,hipBlockIdx_x,flu_kp)];
//
/*
i = hipThreadIdx_x
j = hipBlockIdx_x
k = hipBlockIdx_y 
nx = hipBlockDim_x
ny = hipGridDim_x
nz = hipGridDim_y
*/
__shared__ float flu_s_b3x[3*BLOCK_SIZE];
for (int ii=0; ii<3; ii++)
{
	flu_s_b3x[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]=flu_s_b[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]/2.0;
}
__syncthreads();
//
int flu_imp,flu_imm;
flu_imm=(hipThreadIdx_x+(*flu_nx)-1)%(*flu_nx);
flu_imp=(hipThreadIdx_x+1)%(*flu_nx);
float flu_temp[3];
flu_temp[(1-1)]=flu_s_b3x[a2D_FinC(3,hipBlockDim_x,(1-1),flu_imp)];
flu_temp[(2-1)]=flu_s_jp_b2[hipThreadIdx_x]/2.0;
flu_temp[(3-1)]=flu_s_kp_b3[hipThreadIdx_x]/2.0;
__syncthreads();
for (int ii=0; ii<3; ii++)
{
	flu_s_b3x[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]=flu_s_b3x[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)]+flu_temp[ii];
}
__syncthreads();
//
// --- tvd1 part
//	first mhdflux
float tvd1_u[5], tvd1_b[3];
for (int ii=0; ii<5; ii++)
{
	tvd1_u[ii]=flu_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)];
}
for (int ii=0; ii<3; ii++)
{
	tvd1_b[ii]=flu_s_b3x[a2D_FinC(3,hipBlockDim_x,ii,hipThreadIdx_x)];
}
//
float v[5];
float thread_c;
int thread_n;
thread_n=hipBlockDim_x;
mhdflux(v,&thread_c,tvd1_u,tvd1_b,&thread_n);
//
__shared__ float mhdflux_max[BLOCK_SIZE];
mhdflux_max[hipThreadIdx_x]=thread_c;
__shared__ float s_c[BLOCK_SIZE];
__syncthreads();
if (hipThreadIdx_x==0)
{
float temp_c_max;
temp_c_max=0.0;
for (int i=0; i<BLOCK_SIZE; i++)
{
	if (mhdflux_max[i]>temp_c_max) temp_c_max=mhdflux_max[i];
}
for (int i=0; i<BLOCK_SIZE; i++)
{
	s_c[i]=temp_c_max;
}
}
__syncthreads();
//__shared__ float c;
float c;
c=s_c[hipThreadIdx_x];
//
if (c>0)
{
	for (int ii=0; ii<5; ii++)
	{
		v[ii]=v[ii]/c;
	}
}
// --- tvd1 part 1
float wr[5];
for (int ii=0; ii<5; ii++)
{
	wr[ii]=tvd1_u[ii]+v[ii];
}
float wl[5];
for (int ii=0; ii<5; ii++)
{
	wl[ii]=tvd1_u[ii]-v[ii];
}
float fr[5];
for (int ii=0; ii<5; ii++)
{
	fr[ii]=c*wr[ii];
}
__shared__ float tvd1_s_tmp1[5*BLOCK_SIZE];
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=wl[ii];
}
__syncthreads();
float fl[5];
for (int ii=0; ii<5; ii++)
{
	fl[ii]=c*tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,flu_imp)];
}
__syncthreads();
//
float flux[5];
for (int ii=0; ii<5; ii++)
{
	flux[ii]=(fr[ii]-fl[ii])/2.0;
}
//
__shared__ float tvd1_s_tmp2[5*BLOCK_SIZE];
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=flux[ii];
}
__syncthreads();
float tvd1_u1[5];
for (int ii=0; ii<5; ii++)
{
	tvd1_u1[ii]=tvd1_u[ii]-(flux[ii]-tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,flu_imm)])*(*flu_dt)/2.0;
}
// --- mhdflux part 2
mhdflux(v,&thread_c,tvd1_u1,tvd1_b,&thread_n);
//
mhdflux_max[hipThreadIdx_x]=thread_c;
__syncthreads();
if (hipThreadIdx_x==0)
{
float temp_c_max;
temp_c_max=0.0;
for (int i=0; i<BLOCK_SIZE; i++)
{
        if (mhdflux_max[i]>temp_c_max) temp_c_max=mhdflux_max[i];
}
for (int i=0; i<BLOCK_SIZE; i++)
{
        s_c[i]=temp_c_max;
}
}
__syncthreads();
c=s_c[hipThreadIdx_x];
//
if (c>0)
{
        for (int ii=0; ii<5; ii++)
        {
                v[ii]=v[ii]/c;
        }
}
// --- tvd1 part 2
for (int ii=0; ii<5; ii++)
{
	wr[ii]=tvd1_u1[ii]+v[ii];
}
for (int ii=0; ii<5; ii++)
{
	wl[ii]=tvd1_u1[ii]-v[ii];	
}
for (int ii=0; ii<5; ii++)
{
	fr[ii]=c*wr[ii];
}
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=fr[ii];
}
__syncthreads();
float dfrp[5];
float dfrm[5];
float dfr[5];
for (int ii=0; ii<5; ii++)
{
	dfrp[ii]=(tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,flu_imp)]-fr[ii])/2.0;
}
for (int ii=0; ii<5; ii++)
{
	dfrm[ii]=(fr[ii]-tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,flu_imm)])/2.0;
}
for (int ii=0; ii<5; ii++)
{
	dfr[ii]=0;
}
//
__syncthreads();
for (int ii=0; ii<5; ii++)
{
	if (dfrp[ii]*dfrm[ii]>0) dfr[ii]=2.0*dfrp[ii]*dfrm[ii]/(dfrp[ii]+dfrm[ii]);
	
}
//
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=wl[ii];
}
__syncthreads();
for (int ii=0; ii<5; ii++)
{
	fl[ii]=c*tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,flu_imp)];
}
//
float dflp[5];
float dflm[5];
float dfl[5];
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=fl[ii];
}
__syncthreads();
for (int ii=0; ii<5; ii++)
{
	dflp[ii]=(fl[ii]-tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,flu_imp)])/2.0;
}
for (int ii=0; ii<5; ii++)
{
	dflm[ii]=(tvd1_s_tmp1[a2D_FinC(5,hipBlockDim_x,ii,flu_imm)]-fl[ii])/2.0;
}
for (int ii=0; ii<5; ii++)
{
	dfl[ii]=0;
}
//
__syncthreads();
for (int ii=0; ii<5; ii++)
{
	if (dflp[ii]*dflm[ii]>0) dfl[ii]=2.0*dflp[ii]*dflm[ii]/(dflp[ii]+dflm[ii]);
}
//
for (int ii=0; ii<5; ii++)
{
	flux[ii]=(fr[ii]-fl[ii]+(dfr[ii]-dfl[ii]))/2.0;
}
//
for (int ii=0; ii<5; ii++)
{
	tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=flux[ii];
}
__syncthreads();
for (int ii=0; ii<5; ii++)
{
	flu_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]=flu_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)]-(flux[ii]-tvd1_s_tmp2[a2D_FinC(5,hipBlockDim_x,ii,flu_imm)])*(*flu_dt);
}
for (int ii=0; ii<5; ii++)
{
	flu_u[a4D_FinC(5,hipBlockDim_x,hipGridDim_x,hipGridDim_y,ii,hipThreadIdx_x,hipBlockIdx_x,hipBlockIdx_y)]=flu_s_u[a2D_FinC(5,hipBlockDim_x,ii,hipThreadIdx_x)];
}
// --- end tvd1
return;
}


__device__ void mhdflux(float *mf_v, float *mf_c, float *mf_u, float *mf_b, int *mf_n)
{
float gamma;
gamma=5.0/3.0;
//
float vx;
vx=mf_u[(2-1)]/mf_u[(1-1)];
//
float ps;
ps=(mf_u[(5-1)]-(mf_u[(2-1)]*mf_u[(2-1)]+mf_u[(3-1)]*mf_u[(3-1)]+mf_u[(4-1)]*mf_u[(4-1)])/mf_u[(1-1)]/2.0)*(gamma-1.0)+(2.0-gamma)*(mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)])/2.0;
//
mf_v[(1-1)]=mf_u[(2-1)];
mf_v[(2-1)]=mf_u[(2-1)]*vx+ps-mf_b[(1-1)]*mf_b[(1-1)];
mf_v[(3-1)]=mf_u[(3-1)]*vx-mf_b[(2-1)]*mf_b[(1-1)];
mf_v[(4-1)]=mf_u[(4-1)]*vx-mf_b[(3-1)]*mf_b[(1-1)];
mf_v[(5-1)]=(mf_u[(5-1)]+ps)*vx-mf_b[(1-1)]*(mf_b[(1-1)]*mf_u[(2-1)]+mf_b[(2-1)]*mf_u[(3-1)]+mf_b[(3-1)]*mf_u[(4-1)])/mf_u[(1-1)];
//
float p;
p=ps-(mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)])/2.0;
//
(*mf_c)=fabs(vx)+sqrt(fabs((mf_b[(1-1)]*mf_b[(1-1)]+mf_b[(2-1)]*mf_b[(2-1)]+mf_b[(3-1)]*mf_b[(3-1)]+gamma*p)/mf_u[(1-1)]));
//
return;
}

