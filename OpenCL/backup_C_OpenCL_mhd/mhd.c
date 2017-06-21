//==========================================================
//
//      TVD split MHD code on OpenCL
//
//==========================================================
//
// written December 2009 by Bijia PANG, pangbijia@gmail.com
//      copyright (C) 2009, 2011, Bijia PANG

//      remember always use fortran array in C

#include <stdio.h>
#include <CL/cl.h>

#include "mhd.h"

int main(int argc, char** argv)
{
printf("program start!\n");
// 
    if(setup() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.setup!\n");
//
    if(run() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.run!\n");
//
    if (cleanup() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.cleanup!\n");
//
printf("program finish!\n");
	return EXIT_SUCCESS;
}
