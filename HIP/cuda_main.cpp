#include <stdio.h>
#include <math.h>
#include "cuda.h"
//#include "hip/hip_runtime.h"
#include <pthread.h>
//#include "/usr/src/linux-headers-4.2.0-27/include/linux/list.h"
#include "parameter.h" 
#include "array_definition.h"
//#include "cuda_funclist.h"
//#include "cuda_function.h" 
//#include "cuda_subroutine.h"
#include "hip_funclist.h"
#include "hip_function.h" 
#include "hip_subroutine.h"

//extern "C" void cuda_main(float *h_u, float *h_b, int *h_nx, int *h_ny, int *h_nz)
void cuda_main(float *h_u, float *h_b, int *h_nx, int *h_ny, int *h_nz)
{
//	general info initialization
int Totalthreads = (*h_nx)*(*h_ny)*(*h_nz);
int numThreadsPerBlock = *h_nx;
int numBlocks = Totalthreads/numThreadsPerBlock;
int NumOfU = 5;
int NumOfB = 3;
//	memory size initialization
size_t u_memSize = NumOfU * numBlocks * numThreadsPerBlock * sizeof(float);
size_t b_memSize = NumOfB * numBlocks * numThreadsPerBlock * sizeof(float);
size_t c_memSize = numBlocks * numThreadsPerBlock * sizeof(float);
size_t int_memSize = sizeof(int);
size_t float_memSize = sizeof(float);
//	data on the host
float *h_dt;
//	data on the device
//	hipMalloc
//	for general purpose
float *d_u, *d_b;
hipMalloc( (void **) &d_u, u_memSize );
hipMalloc( (void **) &d_b, b_memSize );
int *d_nx,*d_ny,*d_nz;
hipMalloc( (void **) &d_nx, int_memSize );
hipMalloc( (void **) &d_ny, int_memSize );
hipMalloc( (void **) &d_nz, int_memSize );
float *d_dt;
hipMalloc( (void **) &d_dt, float_memSize );
//	for cuda_cfl
float *d_c;
hipMalloc( (void **) &d_c, c_memSize );
//	for cuda_advectbyzx
float *d_temp;
hipMalloc( (void **) &d_temp, c_memSize );
//	for cuda_transpose
float *d_ut, *d_bt;
hipMalloc( (void **) &d_ut, u_memSize );
hipMalloc( (void **) &d_bt, b_memSize );
//	hipMemcpy
//	copy data from host to device
hipMemcpy( d_u, h_u, u_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_b, h_b, b_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_nx, h_nx, int_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_ny, h_ny, int_memSize, hipMemcpyHostToDevice );
hipMemcpy( d_nz, h_nz, int_memSize, hipMemcpyHostToDevice );
//
//checkCUDAError("memcpy: from host to device, in cuda_main");
//	initialize data for loop
float t,dt,tf;
int iter;
float ct;
printf("in the cuda_main\n");
t=0;
iter=0;
ct=100.;
tf=ct*10;
//	initialization for timing
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);
//	in milliseconds with a resolution of around 0.5 microseconds
float elapsedTime;
float elapsed;
printf("move into to do loop\n");
printf("%f",tf);
extern void cuda_cfl(float *cfl_u, float *cfl_b, int *cfl_nx, int *cfl_ny, int *cfl_nz, float *cfl_dt, float *cfl_c, int *h_cfl_nx, int *h_cfl_ny, int *h_cfl_nz, float *h_cfl_dt);
do {
//	start the timer
	hipEventRecord(start,0);
//	output
//	if you want to output, you have to use hipMemcpy
//	copy the data from device to host to output
	hipMemcpy( h_u, d_u, u_memSize, hipMemcpyDeviceToHost );
	hipMemcpy( h_b, d_b, b_memSize, hipMemcpyDeviceToHost );
        printf("output\n");
	printf("t=	%f,	%i,	%f\n",t,iter,h_u[a4D_FinC(5,(*h_nx),(*h_ny),(*h_nz),(5-1),(*h_nx)/4,1,1)]);
//	done output
        printf("done output\n");
	iter=iter+1;
        printf("into the cuda_cfl\n");
	cuda_cfl(d_u,d_b,d_nx,d_ny,d_nz,d_dt,d_c,h_nx,h_ny,h_nz,h_dt);
        printf("cuda_cfl done\n");
	dt=0.9*(*h_dt);
//	dt=0.5;
	if (dt>(tf-t)/2.0) dt=(tf-t)/2.0;
        printf("dt= %f\n",dt);
	t=t+2.0*dt;
//	start sweep
        printf("cuda_fluidx\n");
	cuda_fluidx(d_u,d_b,d_nx,d_ny,d_nz,d_dt,h_nx,h_ny,h_nz);
        printf("cuda_advectbyzx\n");
	cuda_advectbyzx(d_u,d_b,d_nx,d_ny,d_nz,d_dt,d_temp,h_nx,h_ny,h_nz);
//	the y sweep
        printf("cuda_transpose12\n");
	cuda_transpose12(d_ut,d_bt,d_u,d_b,d_nx,d_ny,d_nz,h_nx,h_ny,h_nz);
	cuda_fluidx(d_u,d_b,d_ny,d_nx,d_nz,d_dt,h_ny,h_nx,h_nz);
	cuda_advectbyzx(d_u,d_b,d_ny,d_nx,d_nz,d_dt,d_temp,h_ny,h_nx,h_nz);
//	z sweep
	cuda_transpose13(d_ut,d_bt,d_u,d_b,d_ny,d_nx,d_nz,h_ny,h_nx,h_nz);
	cuda_fluidx(d_u,d_b,d_nz,d_nx,d_ny,d_dt,h_nz,h_nx,h_ny);
	cuda_advectbyzx(d_u,d_b,d_nz,d_nx,d_ny,d_dt,d_temp,h_nz,h_nx,h_ny);
	cuda_advectbyzx(d_u,d_b,d_nz,d_nx,d_ny,d_dt,d_temp,h_nz,h_nx,h_ny);
	cuda_fluidx(d_u,d_b,d_nz,d_nx,d_ny,d_dt,h_nz,h_nx,h_ny);

//	back
	cuda_transpose13(d_ut,d_bt,d_u,d_b,d_nz,d_nx,d_ny,h_nz,h_nx,h_ny);
	cuda_advectbyzx(d_u,d_b,d_ny,d_nx,d_nz,d_dt,d_temp,h_ny,h_nx,h_nz);
	cuda_fluidx(d_u,d_b,d_ny,d_nx,d_nz,d_dt,h_ny,h_nx,h_nz);
//	x again
	cuda_transpose12(d_ut,d_bt,d_u,d_b,d_ny,d_nx,d_nz,h_ny,h_nx,h_nz);
	cuda_advectbyzx(d_u,d_b,d_nx,d_ny,d_nz,d_dt,d_temp,h_nx,h_ny,h_nz);
	cuda_fluidx(d_u,d_b,d_nx,d_ny,d_nz,d_dt,h_nx,h_ny,h_nz);
//	finish sweep
//	stop the timer
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&elapsedTime,start,stop);	
	printf("time per loop(in milliseconds):	%f\n",elapsedTime);
        elapsed += elapsedTime;
} while (t<tf); //(iter<=100);//(t<tf);
printf("IN AVERAGE, time per loop(in milliseconds): %f\n",elapsed/iter);///100);
printf("%i,%f\n",iter,t);
//
//      hipMemcpy
//      copy data from device to host
hipMemcpy( h_u, d_u, u_memSize, hipMemcpyDeviceToHost );
hipMemcpy( h_b, d_b, b_memSize, hipMemcpyDeviceToHost );
//
//checkCUDAError("memcpy: from device to host, in cuda_main");
//
hipFree(d_u);
hipFree(d_b);
hipFree(d_nx);
hipFree(d_ny);
hipFree(d_nz);
hipFree(d_dt);
//
hipEventDestroy(start);
hipEventDestroy(stop);
//
return;
}

