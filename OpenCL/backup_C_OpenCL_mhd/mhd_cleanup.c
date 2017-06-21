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

//
//	for all functions
int
cleanup_ReleaseKernel()
{
//
    if(cleanup_fluidx_ReleaseKernel() != MHD_SUCCESS)
    {
	printf("Error: cleanup_fluidx_ReleaseKernel in cleanup_ReleaseKernel()!\n");
	return MHD_FAILURE;
    }
//
    if(cleanup_advectbyzx_ReleaseKernel() != MHD_SUCCESS)
    {
	printf("Error: cleanup_advectbyzx_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose12u1234_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12u1234_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
    if(cleanup_transpose12u5b_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12u5b_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose13u1234_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13u1234_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
    if(cleanup_transpose13u5b_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13u5b_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_timing_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_timing_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for fluidx function
int
cleanup_fluidx_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(fluidx_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release fluidx_Kernel!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for advectbyzx function
int
cleanup_advectbyzx_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(advectbyzxA1_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxA1_Kernel!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseKernel(advectbyzxA2_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxA2_Kernel!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseKernel(advectbyzxB1_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxB1_Kernel!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseKernel(advectbyzxB2_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxB2_Kernel!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

int
cleanup_transpose12u1234_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(transpose12u1234_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12u1234_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose12u5b_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(transpose12u5b_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12u5b_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose13u1234_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(transpose13u1234_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13u1234_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose13u5b_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(transpose13u5b_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13u5b_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_timing_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseKernel(timing_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release timing_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

//	for all functions
int
cleanup_ReleaseProgram()
{

    if(cleanup_fluidx_ReleaseProgram() != MHD_SUCCESS)
    {
	printf("Error: cleanup_fluidx_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_advectbyzx_ReleaseProgram() != MHD_SUCCESS)
    {
	printf("Error: cleanup_advectbyzx_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose12u1234_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12u1234_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
    if(cleanup_transpose12u5b_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12u5b_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose13u1234_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13u1234_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
    if(cleanup_transpose13u5b_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13u5b_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_timing_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_timing_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for fluidx function
int
cleanup_fluidx_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(fluidx_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release fluidx_Program!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for advectbyzx function
int
cleanup_advectbyzx_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(advectbyzxA1_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxA1_Program!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseProgram(advectbyzxA2_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxA2_Program!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseProgram(advectbyzxB1_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxB1_Program!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseProgram(advectbyzxB2_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release advectbyzxB2_Program!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose12u1234_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(transpose12u1234_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12u1234_Program!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

int
cleanup_transpose12u5b_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(transpose12u5b_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12u5b_Program!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose13u1234_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(transpose13u1234_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13u1234_Program!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_transpose13u5b_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(transpose13u5b_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13u5b_Program!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup_timing_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;
//
    status_in_cleanup = clReleaseProgram(timing_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release timing_Program!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status_in_cleanup = CL_SUCCESS;
//
    if (cleanup_ReleaseKernel() != MHD_SUCCESS)
    {
	printf("Error: cleanup_ReleaseKernel in cleanup()!\n");
	return MHD_FAILURE;
    }
//
    if (cleanup_ReleaseProgram() != MHD_SUCCESS)
    {
	printf("Error: cleanup_ReleaseProgram in cleanup()!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.u1234);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u1234)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.u5);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u5)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.b);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.b)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.adv_tmpB);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.adv_tmpB)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.u_update1234);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u_update1234)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.u_update5);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u_update5)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseMemObject(ctx_device.b_update);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.b_update)\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseCommandQueue(mhdCommandQueue);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release command queue!\n");
        return MHD_FAILURE;
    }
//
    status_in_cleanup = clReleaseContext(mhdComputeContext);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release compute context!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}
