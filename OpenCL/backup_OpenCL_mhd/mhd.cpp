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
#include <SDKUtil/SDKUtil.hpp>
#include <SDKUtil/SDKFile.hpp>
#include <CL/cl.h>

#include "mhd.hpp"

int main(int argc, char** argv)
{
printf("program start!\n");
// 
    MHD cl_MHD;
//
    if(cl_MHD.setup() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.setup!\n");
//
    if(cl_MHD.run() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.run!\n");
//
    if (cl_MHD.cleanup() != EXIT_SUCCESS)
	return EXIT_FAILURE;
printf("finished cl_MHD.cleanup!\n");
//
printf("program finish!\n");
	return EXIT_SUCCESS;
}
