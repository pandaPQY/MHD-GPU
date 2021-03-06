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

#include "parameter.h" 
#include "array_definition.h"
#include "c_subroutine.h"

int main ()
{
//	general info initialization
variable_context ctx;

float u[nu*total_cell_number];
float b[nb*total_cell_number];
//extern int cuda_main(float *h_u, float *h_b, int *h_nx, int *h_ny, int *h_nz)
//	call init
ctx.u = u;
ctx.b = b;

init(ctx.u,ctx.b,&ctx.nx,&ctx.ny,&ctx.nz);
printf("here?\n");
//	calculation on cuda
cuda_main(ctx.u,ctx.b,&ctx.nx,&ctx.ny,&ctx.nz);
//
return(0);
}
//	

