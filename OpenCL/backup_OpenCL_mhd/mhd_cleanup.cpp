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

//
//	for all functions
int
MHD::cleanup_ReleaseKernel()
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
    if(cleanup_transpose12_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12_ReleaseKernel in cleanup_ReleaseKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose13_ReleaseKernel() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13_ReleaseKernel in cleanup_ReleaseKernel()!\n");
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
MHD::cleanup_fluidx_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseKernel(fluidx_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release fluidx_Kernel!\n");
        return MHD_FAILURE;
    }

	return MHD_SUCCESS;
}

//	for advectbyzx function
int
MHD::cleanup_advectbyzx_ReleaseKernel()
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
MHD::cleanup_transpose12_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseKernel(transpose12_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12_Kernel!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::cleanup_transpose13_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseKernel(transpose13_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13_Kernel!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::cleanup_timing_ReleaseKernel()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseKernel(timing_Kernel);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release timing_Kernel!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

//	for all functions
int
MHD::cleanup_ReleaseProgram()
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
    if(cleanup_transpose12_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose12_ReleaseProgram in cleanup_ReleaseProgram()!\n");
        return MHD_FAILURE;
    }
//
    if(cleanup_transpose13_ReleaseProgram() != MHD_SUCCESS)
    {
        printf("Error: cleanup_transpose13_ReleaseProgram in cleanup_ReleaseProgram()!\n");
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
MHD::cleanup_fluidx_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseProgram(fluidx_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release fluidx_Program!\n");
        return MHD_FAILURE;
    }

	return MHD_SUCCESS;
}

//	for advectbyzx function
int
MHD::cleanup_advectbyzx_ReleaseProgram()
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
MHD::cleanup_transpose12_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseProgram(transpose12_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose12_Program!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::cleanup_transpose13_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseProgram(transpose13_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release transpose13_Program!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::cleanup_timing_ReleaseProgram()
{
    cl_int status_in_cleanup = CL_SUCCESS;

    status_in_cleanup = clReleaseProgram(timing_Program);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release timing_Program!\n");
        return MHD_FAILURE;
    }

        return MHD_SUCCESS;
}

int
MHD::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status_in_cleanup = CL_SUCCESS;

    if (cleanup_ReleaseKernel() != MHD_SUCCESS)
    {
	printf("Error: cleanup_ReleaseKernel in cleanup()!\n");
	return MHD_FAILURE;
    }

    if (cleanup_ReleaseProgram() != MHD_SUCCESS)
    {
	printf("Error: cleanup_ReleaseProgram in cleanup()!\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseMemObject(ctx_device.u);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u)\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseMemObject(ctx_device.b);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.b)\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseMemObject(ctx_device.adv_tmpB);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.adv_tmpB)\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseMemObject(ctx_device.u_update);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.u_update)\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseMemObject(ctx_device.b_update);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release device memory! (ctx_device.b_update)\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseCommandQueue(mhdCommandQueue);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release command queue!\n");
        return MHD_FAILURE;
    }

    status_in_cleanup = clReleaseContext(mhdComputeContext);
    if (status_in_cleanup != CL_SUCCESS)
    {
        printf("Error: Failed to release compute context!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}
