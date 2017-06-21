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
int
MHD::setupMHD_initialization()
{

    WorkGroupSize = SIZEofWORKGROUP;
//
    ctx_host.u = (data_type_t*)memalign(16,total_cell_number*nu*sizeof(data_type_t));
    if(ctx_host.u == NULL)
    {
	printf("Error: Failed to allocate ctx_host.u host memory!\n");    
        return MHD_FAILURE;
    }
//
    ctx_host.b = (data_type_t*)memalign(16,total_cell_number*nb*sizeof(data_type_t));
    if(ctx_host.b == NULL)
    {
	printf("Error: Failed to allocate ctx_host.b host memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_host.adv_tmpB = (data_type_t*)memalign(16,total_cell_number*sizeof(data_type_t));
    if(ctx_host.adv_tmpB == NULL)
    {
        printf("Error: Failed to allocate ctx_host.adv_tmpB host memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_host.u_update = (data_type_t*)memalign(16,total_cell_number*nu*sizeof(data_type_t));
    if(ctx_host.u_update == NULL)
    {
        printf("Error: Failed to allocate ctx_host.u_update host memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_host.b_update = (data_type_t*)memalign(16,total_cell_number*nb*sizeof(data_type_t));
    if(ctx_host.b_update == NULL)
    {
        printf("Error: Failed to allocate ctx_host.b_update host memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_host.cfl_tmpC = (data_type_t*)memalign(16,box_ny*box_nz*sizeof(data_type_t));
    if(ctx_host.cfl_tmpC == NULL)
    {
        printf("Error: Failed to allocate ctx_host.cfl_tmpC host memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_host.fluidx_test = (int*)memalign(16,total_cell_number*sizeof(int));
    if(ctx_host.fluidx_test == NULL)
    {
        printf("Error: Failed to allocate ctx_host.fluidx_test host memory!\n");
        return MHD_FAILURE;
    }
//
int i,j,k;
for (k=0;k<(box_nz);k++)
{
        for (j=0;j<(box_ny);j++)
        {
                for (i=0;i<(box_nx);i++)
                {
ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)]=1;
//printf("i,j,k,test value: %i,%i,%i,%i\n",i,j,k,ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)]);
                }
        }
}

//
    MHD::init_value(ctx_host.u,ctx_host.b,&ctx_host.nx,&ctx_host.ny,&ctx_host.nz,ctx_host.adv_tmpB,ctx_host.u_update,ctx_host.b_update);
// 
    ctx_device.nx=box_nx;
    ctx_device.ny=box_ny;
    ctx_device.nz=box_nz;
//
	return MHD_SUCCESS;
}

void 
MHD::init_value(data_type_t *initV_u, data_type_t *initV_b, int *initV_nx, int *initV_ny, int *initV_nz, data_type_t *initV_adv_tmpB, data_type_t *initV_u_update, data_type_t *initV_b_update)
{
printf("beginning in init\n");

  *initV_nx = box_nx;
  *initV_ny = box_ny;
  *initV_nz = box_nz;

  int i,j,k,ii;
  FILE *initV_File;
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvenlinear_16cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvenlinear_64cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvenlinear_128cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvenlinear_32cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvencircular_64cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvencircular_128cube.dat","r+");
  //initV_File=fopen("/cita/h/home-1/bpang/cuda_MHD/gogogo/sep29_first_fluidx/data/source_init_alfvencircular_32cube.dat","r+");
  initV_File=fopen("source_init_alfvenlinear_128cube.dat","r+");
  for (i=0;i<(*initV_nx);i++)
  {
        for (j=0;j<(*initV_ny);j++)
        {
                for (k=0;k<(*initV_nz);k++)
                {
                        initV_adv_tmpB[a3D_FinC((*initV_nx),(*initV_ny),(*initV_nz),i,j,k)]=0.0E0;
for (ii=0;ii<5;ii++)
{
                        initV_u_update[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),ii,i,j,k)]=0.0E0;
}
for (ii=0;ii<3;ii++)
{
                        initV_b_update[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),ii,i,j,k)]=0.0E0;
}

                        if (sizeof(data_type_t)==4)
                        {
/*
                        fscanf(initV_File,"%f",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),0,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),1,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),2,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),3,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),4,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),0,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),1,i,j,k)]);
                        fscanf(initV_File,"%f",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),2,i,j,k)]);
*/
                        fread(&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),(1-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),(2-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),(3-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),(4-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),(5-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),(1-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),(2-1),i,j,k)],sizeof(float),1,initV_File);
                        fread(&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),(3-1),i,j,k)],sizeof(float),1,initV_File);
                        }
                        else if (sizeof(data_type_t)==8)
                        {
                        fscanf(initV_File,"%lf",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),0,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),1,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),2,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),3,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_u[a4D_FinC(5,(*initV_nx),(*initV_ny),(*initV_nz),4,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),0,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),1,i,j,k)]);
                        fscanf(initV_File,"%lf",&initV_b[a4D_FinC(3,(*initV_nx),(*initV_ny),(*initV_nz),2,i,j,k)]);
                        }
                }
        }
  }
  fclose(initV_File);

printf("t= %e,  %e,   %e\n",initV_b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(1-1),(5-1),(5-1),(1-1))],initV_b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(2-1),(5-1),(5-1),(1-1))],initV_b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(3-1),(5-1),(5-1),(1-1))]);

printf("end in init\n");
}

int
MHD::setupCL_ContextCommandMemory()
{
    cl_int status_in_setupCL = CL_SUCCESS;

    //mhdDeviceType = CL_DEVICE_TYPE_CPU;//CL_DEVICE_TYPE_GPU;//CL_DEVICE_TYPE_CPU;
    mhdDeviceType = CL_DEVICE_TYPE_GPU;//CL_DEVICE_TYPE_CPU;

 cl_platform_id platform;
 status_in_setupCL=clGetPlatformIDs(1,&platform,NULL) ;
 cl_device_id device ;
 status_in_setupCL=clGetDeviceIDs(platform , CL_DEVICE_TYPE_ALL , 1 , &device ,NULL) ;
 mhdComputeContext=clCreateContext (NULL, 1 , &device , NULL, NULL,&status_in_setupCL ) ;
    /* Create context from given device type */
/*
    mhdComputeContext = clCreateContextFromType(
		0,
                mhdDeviceType,
                NULL,
                NULL,
                &status_in_setupCL);
*/
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create a compute context!\n");
        return MHD_FAILURE;
    }

    /* First, get the size of device list data */
    size_t deviceListSize;
    status_in_setupCL = clGetContextInfo(
		mhdComputeContext,
                CL_CONTEXT_DEVICES,
                0,
                NULL,
                &deviceListSize);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get the size of device list!\n");
        return MHD_FAILURE;
    }

    /* Now allocate memory for device list based on the size we got earlier */
    mhdComputeDevice = (cl_device_id *)malloc(deviceListSize);
    if(mhdComputeDevice == NULL) 
    {
	printf("Error: Failed to allocate memory for device list!\n");
        return MHD_FAILURE;
    }

    /* Now, get the device list data */
    status_in_setupCL = clGetContextInfo(
		mhdComputeContext,
                CL_CONTEXT_DEVICES,
                deviceListSize,
                mhdComputeDevice,
                NULL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get device list data!\n");
        return MHD_FAILURE;
    }

    /* Create command queue */
    mhdProp = 0;
    if(TimeTheTime)
	mhdProp |= CL_QUEUE_PROFILING_ENABLE;

    mhdCommandQueue = clCreateCommandQueue(
		mhdComputeContext,
                mhdComputeDevice[0],
                mhdProp,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to create command queue!\n");
        return MHD_FAILURE;
    }

    /* Get Device specific Information */
    status_in_setupCL = clGetDeviceInfo(
		mhdComputeDevice[0],
		CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(size_t),
                (void*)&mhdMaxWorkGroupSize,
                NULL);
printf("Max Work Group Size is %i\n",mhdMaxWorkGroupSize);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get CL_DEVICE_MAX_WORK_GROUP_SIZE!\n");
        return MHD_FAILURE;
    }

    status_in_setupCL = clGetDeviceInfo(
		mhdComputeDevice[0],
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                sizeof(cl_uint),
                (void*)&mhdMaxDimensions,
                NULL);
printf("Max Dimensions is %i\n",mhdMaxDimensions);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS!\n");
        return MHD_FAILURE;
    }

    mhdMaxWorkItemSizes = (size_t*)malloc(mhdMaxDimensions * sizeof(size_t));
    status_in_setupCL = clGetDeviceInfo(
		mhdComputeDevice[0],
                CL_DEVICE_MAX_WORK_ITEM_SIZES,
                sizeof(size_t) * mhdMaxDimensions,
                (void*)mhdMaxWorkItemSizes,
                NULL);
printf("Max Work Item Sizes is %i	%i	%i\n",mhdMaxWorkItemSizes[0],mhdMaxWorkItemSizes[1],mhdMaxWorkItemSizes[2]);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get CL_DEVICE_MAX_WORK_ITEM_SIZES!\n");
        return MHD_FAILURE;
    }

    status_in_setupCL = clGetDeviceInfo(
		mhdComputeDevice[0],
                CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(cl_ulong),
                (void *)&mhdTotalLocalMemory,
                NULL);
printf("Total Local Memory is %i\n",mhdTotalLocalMemory);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to get CL_DEVICE_LOCAL_MEM_SIZE!\n");
        return MHD_FAILURE;
    }

    /*
 *      * Create and initialize memory objects
 *           */

    /* Create memory objects */
//	fluid component
    ctx_device.u = clCreateBuffer(
		mhdComputeContext,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		total_cell_number*nu*sizeof(data_type_t),	
		ctx_host.u,
		&status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to allocate fluid component in device memory!\n");
        return MHD_FAILURE;
    }
//	magnetic component
    ctx_device.b = clCreateBuffer(
		mhdComputeContext,
		CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		total_cell_number*nb*sizeof(data_type_t),
		ctx_host.b,
		&status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
	printf("Error: Failed to allocate magnetic component in device memory!\n");
        return MHD_FAILURE;
    }
//	adv_tmpB
    ctx_device.adv_tmpB = clCreateBuffer(
                mhdComputeContext,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                total_cell_number*sizeof(data_type_t),
                ctx_host.adv_tmpB,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to allocate adv for temporary in device memory!\n");
        return MHD_FAILURE;
    }
//	u_update
    ctx_device.u_update = clCreateBuffer(
                mhdComputeContext,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                total_cell_number*nu*sizeof(data_type_t),
                ctx_host.u_update,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to allocate fluid update component in device memory!\n");
        return MHD_FAILURE;
    }
//	b_update
    ctx_device.b_update = clCreateBuffer(
                mhdComputeContext,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                total_cell_number*nb*sizeof(data_type_t),
                ctx_host.b_update,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to allocate magnetic update component in device memory!\n");
        return MHD_FAILURE;
    }
//	cfl_tmpC
    ctx_device.cfl_tmpC = clCreateBuffer(
                mhdComputeContext,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                box_ny*box_nz*sizeof(data_type_t),
                ctx_host.cfl_tmpC,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to allocate cfl for temporary in device memory!\n");
        return MHD_FAILURE;
    }
//
    ctx_device.fluidx_test = clCreateBuffer(
                mhdComputeContext,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                total_cell_number*sizeof(int),
                ctx_host.fluidx_test,
                &status_in_setupCL);
    if (status_in_setupCL != CL_SUCCESS)
    {
        printf("Error: Failed to allocate fluidx for test in device memory!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for all functions
int
MHD::setupCL_ProgramKernel()
{

    if(setupCL_calcfl_ProgramKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_calcfl_ProgramKernel in setupCL_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//

    if(setupCL_fluidx_ProgramKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_fluidx_ProgramKernel in setupCL_ProgramKernel()!\n");
	return MHD_FAILURE;
    }

//
    if(setupCL_advectbyzx_ProgramKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_advectbyzx_ProgramKernel in setupCL_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_transpose12_ProgramKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_transpose12_ProgramKernel in setupCL_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_transpose13_ProgramKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_transpose13_ProgramKernel in setupCL_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
if (TimeTheTime)
{
    if(setupCL_timing_ProgramKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_timing_ProgramKernel in setupCL_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
}
	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzx_ProgramKernel()
{
//
    if(setupCL_advectbyzxA1_ProgramKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxA1_ProgramKernel in setupCL_advectbyzx_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxA2_ProgramKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxA2_ProgramKernel in setupCL_advectbyzx_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxB1_ProgramKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxB1_ProgramKernel in setupCL_advectbyzx_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxB2_ProgramKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxB2_ProgramKernel in setupCL_advectbyzx_ProgramKernel()!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	for all functions
int
MHD::setupCL_ArgumentKernel()
{
//
    if(setupCL_calcfl_ArgumentKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_calcfl_ArgumentKernel in setupCL_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//

    if(setupCL_fluidx_ArgumentKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_fluidx_ArgumentKernel in setupCL_ArgumentKernel()!\n");
	return MHD_FAILURE;
    }

//
    if(setupCL_advectbyzx_ArgumentKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_advectbyzx_ArgumentKernel in setupCL_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_transpose12_ArgumentKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_transpose12_ArgumentKernel in setupCL_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_transpose13_ArgumentKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_transpose13_ArgumentKernel in setupCL_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
if (TimeTheTime)
{
    if(setupCL_timing_ArgumentKernel() != MHD_SUCCESS)
    {
        printf("Error: setupCL_timing_ArgumentKernel in setupCL_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
}
	return MHD_SUCCESS;
}

int
MHD::setupCL_advectbyzx_ArgumentKernel()
{
//
    if(setupCL_advectbyzxA1_ArgumentKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxA1_ArgumentKernel in setupCL_advectbyzx_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxA2_ArgumentKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxA2_ArgumentKernel in setupCL_advectbyzx_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxB1_ArgumentKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxB1_ArgumentKernel in setupCL_advectbyzx_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
    if(setupCL_advectbyzxB2_ArgumentKernel() != MHD_SUCCESS)
    {
printf("Error: setupCL_advectbyzxB2_ArgumentKernel in setupCL_advectbyzx_ArgumentKernel()!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}


int 
MHD::setup()
{
//
    if(setupMHD_initialization() != MHD_SUCCESS)
    {
	printf("Error: setupMHD_initialization in setup()!\n");
	return MHD_FAILURE;
    }
//
    if(setupCL_ContextCommandMemory() != MHD_SUCCESS)
    {
	printf("Error: setupCL_ContextCommandMemory in setup()!\n");
	return MHD_FAILURE;
    }
//
    if(setupCL_ProgramKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_ProgramKernel in setup()!\n");
	return MHD_FAILURE;
    }
//
    if(setupCL_ArgumentKernel() != MHD_SUCCESS)
    {
	printf("Error: setupCL_ArgumentKernel in setup()!\n");
	return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}
