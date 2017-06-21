#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <CL/cl.h>

#include "data_type.h"
#include "struct_def.h"
#include "info_def.h"
#include "parameter.h"
#include "array_definition.h"
#include "mhd.h"

int
setupCL_calcfl_ProgramKernel()
{

    cl_int status_in_setupCL = CL_SUCCESS;

    /* create a CL program using the kernel source */
    int number_of_file = 4;
    FILE* file[number_of_file];
    char *string_mhd[number_of_file];
    const char* source[number_of_file];
    size_t sourceSize[number_of_file];
    file[0]=fopen("data_type.h","rb");
    file[1]=fopen("parameter.h","rb");
    file[2]=fopen("array_definition.h","rb");
    file[3]=fopen("calcfl_Kernels.cl","rb");
    int i;
    for (i=0; i<number_of_file; i++)
    {
	if (file[i] == NULL)
	{
		printf("can't open file\n");
		return MHD_FAILURE;
	}
	fseek(file[i], 0, SEEK_END);
	sourceSize[i] = ftell(file[i]);
	fseek(file[i], 0, SEEK_SET);
	string_mhd[i] = (char*) malloc (sourceSize[i]+1);
	fread(string_mhd[i],sizeof(char),sourceSize[i],file[i]);	
	fclose (file[i]);
        source[i] = (const char *)(string_mhd[i]);
    }
    calcfl_Program = clCreateProgramWithSource(
		mhdComputeContext,
	        number_of_file,
	        &source[0],
	        sourceSize,
	        &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create program for calcfl_Kernels.cl!\n");
        return MHD_FAILURE;
    }

//const char *header = "-I /usr/include/math.h\0";
//err = clBuildProgram(program, 1, devices, header, NULL, NULL); 
    /* create a cl program executable for all the devices specified */
    status_in_setupCL = clBuildProgram(
		calcfl_Program,
                1,
                &mhdComputeDevice[0],
               // header,
                NULL,
                NULL,
                NULL);
//sprintf(buffer,"%s -I %s",OPENCL_OPTIONS,buf2);
//int cl_build_err=clBuildProgram(clprog, 0, NULL, buffer, NULL, NULL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	char buffer[2048];
	size_t len;
	clGetProgramBuildInfo(calcfl_Program, mhdComputeDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	printf("%s\n", buffer);
	printf("Error: Failed to build program executable for calcfl_Kernels.cl!\n");
	return MHD_FAILURE;
    }

    /* get a kernel object handle for a kernel with the given name */
    calcfl_Kernel = clCreateKernel(
		calcfl_Program,
                "calcfl",
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create kernel for calcfl_Kernels.cl!\n");
	return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
setupCL_calcfl_ArgumentKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;

    /*** Set appropriate arguments to the kernel ***/

//	u, fluid component
    status_in_setupCL = clSetKernelArg(
		calcfl_Kernel,
                0,
                sizeof(cl_mem),
                (void *)&ctx_device.u1234);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to set argument to the kernel! (u1234)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                1,
                sizeof(cl_mem),
                (void *)&ctx_device.u5);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u5)\n");
        return MHD_FAILURE;
    }
//	b, magnetic component
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                2,
                sizeof(cl_mem),
                (void *)&ctx_device.b);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b)\n");
        return MHD_FAILURE;
    }
//	nx, X length
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                3,
                sizeof(cl_int),
                (void *)&ctx_device.nx);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nx)\n");
        return MHD_FAILURE;
    }
//	ny, Y length
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                4,
                sizeof(cl_int),
                (void *)&ctx_device.ny);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (ny)\n");
        return MHD_FAILURE;
    }
//	nz, Z length
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                5,
                sizeof(cl_int),
                (void *)&ctx_device.nz);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nz)\n");
        return MHD_FAILURE;
    }
//	dt, cfl time 
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                6,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
//	cfl_tmpC, matrix for cfl
    status_in_setupCL = clSetKernelArg(
                calcfl_Kernel,
                7,
                sizeof(cl_mem),
                (void *)&ctx_device.cfl_tmpC);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (cfl_tmpC)\n");
        return MHD_FAILURE;
    }
//	do the memory check
    status_in_setupCL = clGetKernelWorkGroupInfo(
		calcfl_Kernel,
		mhdComputeDevice[0],
                CL_KERNEL_LOCAL_MEM_SIZE,
                sizeof(cl_ulong),
                &mhdUsedLocalMemory,
                NULL);
printf("mhdUsedLocalMemory is %i\n",mhdUsedLocalMemory);
printf("mhdTotalLocalMemory is %i\n",mhdTotalLocalMemory);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get CL_KERNEL_LOCAL_MEM_SIZE!\n");
        return MHD_FAILURE;
    }
    if(mhdUsedLocalMemory > mhdTotalLocalMemory)
    {
        printf("Unsupported: Insufficient local memory on device!\n");
        return MHD_FAILURE;
    }
//	get the other parameters about the kernel
    status_in_setupCL = clGetKernelWorkGroupInfo(
                calcfl_Kernel,
                mhdComputeDevice[0],
                CL_KERNEL_WORK_GROUP_SIZE,
                sizeof(size_t),
                &mhdKernelWorkGroupSize,
                NULL);
printf("mhdKernelWorkGroupSize is %i\n",mhdKernelWorkGroupSize);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to get CL_KERNEL_WORK_GROUP_SIZE!\n");
        return MHD_FAILURE;
    }
//
		return MHD_SUCCESS;
}
