#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <SDKUtil/SDKUtil.hpp>
#include <SDKUtil/SDKFile.hpp>
#include <CL/cl.h>

#include "data_type.hpp"
#include "struct_def.hpp"
#include "info_def.hpp"
#include "parameter.hpp"
#include "array_definition.hpp"
#include "mhd.hpp"

int
MHD::setupCL_advectbyzxA1_ProgramKernel()
{

    cl_int status_in_setupCL = CL_SUCCESS;

    /* create a CL program using the kernel source */
    int number_of_file = 4;
    appsdk::SDKFile kernelFile[number_of_file];
    kernelFile[0].open("data_type.hpp");
    kernelFile[1].open("parameter.hpp");
    kernelFile[2].open("array_definition.hpp");
    kernelFile[3].open("advectbyzxA1_Kernels.cl");
    int i;
    const char * source[number_of_file];
    size_t sourceSize[number_of_file];
    for (i=0; i<number_of_file; i++)
    { 
	source[i] = kernelFile[i].source().c_str();
	sourceSize[i] = strlen(source[i]);
    }
//
    advectbyzxA1_Program = clCreateProgramWithSource(
		mhdComputeContext,
	        number_of_file,
	        &source[0],
	        sourceSize,
	        &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create program for advectbyzxA1_Kernels.cl!\n");
        return MHD_FAILURE;
    }

//const char *header = "-I /usr/include/math.h\0";
//err = clBuildProgram(program, 1, devices, header, NULL, NULL); 
    /* create a cl program executable for all the devices specified */
    status_in_setupCL = clBuildProgram(
		advectbyzxA1_Program,
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
	clGetProgramBuildInfo(advectbyzxA1_Program, mhdComputeDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
	printf("%s\n", buffer);
	printf("Error: Failed to build program executable for advectbyzxA1_Kernels.cl!\n");
	return MHD_FAILURE;
    }

    /* get a kernel object handle for a kernel with the given name */
    advectbyzxA1_Kernel = clCreateKernel(
		advectbyzxA1_Program,
                "advectbyzxA1",
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create kernel for advectbyzxA1_Kernels.cl!\n");
	return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxA2_ProgramKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    int number_of_file = 4;
    appsdk::SDKFile kernelFile[number_of_file];
    kernelFile[0].open("data_type.hpp");
    kernelFile[1].open("parameter.hpp");
    kernelFile[2].open("array_definition.hpp");
    kernelFile[3].open("advectbyzxA2_Kernels.cl");
    int i;
    const char * source[number_of_file];
    size_t sourceSize[number_of_file];
    for (i=0; i<number_of_file; i++)
    {
        source[i] = kernelFile[i].source().c_str();
        sourceSize[i] = strlen(source[i]);
    }
//
    advectbyzxA2_Program = clCreateProgramWithSource(
                mhdComputeContext,
                number_of_file,
                &source[0],
                sourceSize,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create program for advectbyzxA2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    status_in_setupCL = clBuildProgram(
                advectbyzxA2_Program,
                1,
                &mhdComputeDevice[0],
                NULL,
                NULL,
                NULL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        char buffer[2048];
        size_t len;
        clGetProgramBuildInfo(advectbyzxA2_Program, mhdComputeDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error: Failed to build program executable for advectbyzxA2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    advectbyzxA2_Kernel = clCreateKernel(
                advectbyzxA2_Program,
                "advectbyzxA2",
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create kernel for advectbyzxA2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxB1_ProgramKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    int number_of_file = 4;
    appsdk::SDKFile kernelFile[number_of_file];
    kernelFile[0].open("data_type.hpp");
    kernelFile[1].open("parameter.hpp");
    kernelFile[2].open("array_definition.hpp");
    kernelFile[3].open("advectbyzxB1_Kernels.cl");
    int i;
    const char * source[number_of_file];
    size_t sourceSize[number_of_file];
    for (i=0; i<number_of_file; i++)
    {
        source[i] = kernelFile[i].source().c_str();
        sourceSize[i] = strlen(source[i]);
    }
//
    advectbyzxB1_Program = clCreateProgramWithSource(
                mhdComputeContext,
                number_of_file,
                &source[0],
                sourceSize,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create program for advectbyzxB1_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    status_in_setupCL = clBuildProgram(
                advectbyzxB1_Program,
                1,
                &mhdComputeDevice[0],
                NULL,
                NULL,
                NULL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        char buffer[2048];
        size_t len;
        clGetProgramBuildInfo(advectbyzxB1_Program, mhdComputeDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error: Failed to build program executable for advectbyzxB1_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    advectbyzxB1_Kernel = clCreateKernel(
                advectbyzxB1_Program,
                "advectbyzxB1",
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create kernel for advectbyzxB1_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxB2_ProgramKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    int number_of_file = 4;
    appsdk::SDKFile kernelFile[number_of_file];
    kernelFile[0].open("data_type.hpp");
    kernelFile[1].open("parameter.hpp");
    kernelFile[2].open("array_definition.hpp");
    kernelFile[3].open("advectbyzxB2_Kernels.cl");
    int i;
    const char * source[number_of_file];
    size_t sourceSize[number_of_file];
    for (i=0; i<number_of_file; i++)
    {
        source[i] = kernelFile[i].source().c_str();
        sourceSize[i] = strlen(source[i]);
    }
//
    advectbyzxB2_Program = clCreateProgramWithSource(
                mhdComputeContext,
                number_of_file,
                &source[0],
                sourceSize,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create program for advectbyzxB2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    status_in_setupCL = clBuildProgram(
                advectbyzxB2_Program,
                1,
                &mhdComputeDevice[0],
                NULL,
                NULL,
                NULL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        char buffer[2048];
        size_t len;
        clGetProgramBuildInfo(advectbyzxB2_Program, mhdComputeDevice[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error: Failed to build program executable for advectbyzxB2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
    advectbyzxB2_Kernel = clCreateKernel(
                advectbyzxB2_Program,
                "advectbyzxB2",
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to create kernel for advectbyzxB2_Kernels.cl!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxA1_ArgumentKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;

    /*** Set appropriate arguments to the kernel ***/

    status_in_setupCL = clSetKernelArg(
		advectbyzxA1_Kernel,
                0,
                sizeof(cl_mem),
                (void *)&ctx_device.u);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to set argument to the kernel! (u)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                1,
                sizeof(cl_mem),
                (void *)&ctx_device.b);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                2,
                sizeof(cl_int),
                (void *)&ctx_device.nx);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nx)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                3,
                sizeof(cl_int),
                (void *)&ctx_device.ny);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (ny)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                4,
                sizeof(cl_int),
                (void *)&ctx_device.nz);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nz)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                6,
                sizeof(cl_mem),
                (void *)&ctx_device.adv_tmpB);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (adv_tmpB)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                7,
                sizeof(cl_mem),
                (void *)&ctx_device.u_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u_update)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                8,
                sizeof(cl_mem),
                (void *)&ctx_device.b_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b_update)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                9,
                sizeof(cl_mem),
                (void *)&ctx_device.fluidx_test);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (fluidx_test)\n");
        return MHD_FAILURE;
    }

//	do the memory check
    status_in_setupCL = clGetKernelWorkGroupInfo(
		advectbyzxA1_Kernel,
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
                advectbyzxA1_Kernel,
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

int
MHD::setupCL_advectbyzxA2_ArgumentKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                0,
                sizeof(cl_mem),
                (void *)&ctx_device.u);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                1,
                sizeof(cl_mem),
                (void *)&ctx_device.b);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                2,
                sizeof(cl_int),
                (void *)&ctx_device.nx);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nx)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                3,
                sizeof(cl_int),
                (void *)&ctx_device.ny);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (ny)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                4,
                sizeof(cl_int),
                (void *)&ctx_device.nz);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nz)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                6,
                sizeof(cl_mem),
                (void *)&ctx_device.adv_tmpB);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (adv_tmpB)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                7,
                sizeof(cl_mem),
                (void *)&ctx_device.u_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u_update)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                8,
                sizeof(cl_mem),
                (void *)&ctx_device.b_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b_update)\n");
        return MHD_FAILURE;
    }

    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxA2_Kernel,
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

    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxA2_Kernel,
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

	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxB1_ArgumentKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                0,
                sizeof(cl_mem),
                (void *)&ctx_device.u);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                1,
                sizeof(cl_mem),
                (void *)&ctx_device.b);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                2,
                sizeof(cl_int),
                (void *)&ctx_device.nx);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nx)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                3,
                sizeof(cl_int),
                (void *)&ctx_device.ny);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (ny)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                4,
                sizeof(cl_int),
                (void *)&ctx_device.nz);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nz)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                6,
                sizeof(cl_mem),
                (void *)&ctx_device.adv_tmpB);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (adv_tmpB)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                7,
                sizeof(cl_mem),
                (void *)&ctx_device.u_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u_update)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                8,
                sizeof(cl_mem),
                (void *)&ctx_device.b_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b_update)\n");
        return MHD_FAILURE;
    }

    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxB1_Kernel,
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
    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxB1_Kernel,
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

	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzxB2_ArgumentKernel()
{
    cl_int status_in_setupCL = CL_SUCCESS;
//
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                0,
                sizeof(cl_mem),
                (void *)&ctx_device.u);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                1,
                sizeof(cl_mem),
                (void *)&ctx_device.b);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                2,
                sizeof(cl_int),
                (void *)&ctx_device.nx);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nx)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                3,
                sizeof(cl_int),
                (void *)&ctx_device.ny);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (ny)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                4,
                sizeof(cl_int),
                (void *)&ctx_device.nz);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (nz)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                6,
                sizeof(cl_mem),
                (void *)&ctx_device.adv_tmpB);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (adv_tmpB)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                7,
                sizeof(cl_mem),
                (void *)&ctx_device.u_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (u_update)\n");
        return MHD_FAILURE;
    }
    status_in_setupCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                8,
                sizeof(cl_mem),
                (void *)&ctx_device.b_update);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (b_update)\n");
        return MHD_FAILURE;
    }

    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxB2_Kernel,
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

    status_in_setupCL = clGetKernelWorkGroupInfo(
                advectbyzxB2_Kernel,
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

	return MHD_SUCCESS;
}

