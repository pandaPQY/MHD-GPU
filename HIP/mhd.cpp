//==========================================================
//
//      TVD split MHD code on CUDA
//
//==========================================================
//
// written October 2009 by Bijia PANG, pangbijia@gmail.com 
//      copyright (C) 2009, 2011, Bijia PANG

//	remember always use fortran array in C
#include <stdio.h>
#include <math.h>
//#include <pthread.h>
#include "parameter.h" 
#include "array_definition.h"
#include "c_subroutine.h"
#include <vector>
#include <stdlib.h>
//struct variable_context {
//  float *u;
//  float *b;
//  int nx;
//  int ny;
//  int nz;
//  float dt;
//};

int main ()
{
printf("begin\n");
//	general info initialization
//float u[nu*total_cell_number];
//float b[nb*total_cell_number];
//float * u = (float *) malloc(nu*total_cell_number * sizeof(float));
//float * b = (float *) malloc(nb*total_cell_number * sizeof(float));
//s=malloc(sizeof(*u));
//std::vector<float> u(nu*total_cell_number);
//std::vector<float> b(nb*total_cell_number);
float *u=new float[nu*total_cell_number];
float *b=new float[nb*total_cell_number];
//u* heapArray = new u[nu*total_cell_number];
//b* heapArray = new b[nu*total_cell_number];
//struct ctx {
//  float *u;
//  float *b;
//  int nx;
//  int ny;
//  int nz;
//  float dt;
//};
variable_context ctx;//=new float[(nu+nb)*(total_cell_number+1)+1];
extern void cuda_main(float *h_u, float *h_b, int *h_nx, int *h_ny, int *h_nz);
//extern void init(float *init_u, float *init_b, int *init_nx, int *init_ny, int *init_nz)
//	call init
printf("start\n");
//ctx.u = new float[nu*total_cell_number];
//ctx.b = new float[nb*total_cell_number];
ctx.u=u;
ctx.b=b;

init(ctx.u,ctx.b,&ctx.nx,&ctx.ny,&ctx.nz);
printf("here?\n");
//	calculation on cuda
cuda_main(ctx.u,ctx.b,&ctx.nx,&ctx.ny,&ctx.nz);
//
delete [] u;
delete [] b;
//delete [] ctx.u;
//delete [] ctx.b;
return(0);
}
//	

