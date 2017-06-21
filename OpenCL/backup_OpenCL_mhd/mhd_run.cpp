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
    size_t globalThreads[2] = {box_nx*box_ny,box_nz};
    size_t localThreads[2] = {box_nx,1};
    double diffTime_all;
//
int
MHD::runCL_UpdateCFLTime()
{
    cl_int status_in_runCL;

//	update fluidx kernel
    status_in_runCL = clSetKernelArg(
                fluidx_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }

//	update advectbyzx 
    status_in_runCL = clSetKernelArg(
                advectbyzxA1_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
//
    status_in_runCL = clSetKernelArg(
                advectbyzxA2_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
//
    status_in_runCL = clSetKernelArg(
                advectbyzxB1_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
//
    status_in_runCL = clSetKernelArg(
                advectbyzxB2_Kernel,
                5,
                sizeof(cl_float),
                (void *)&ctx_device.dt);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to set argument to the kernel! (dt)\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

//	Read the buffer from device to host
int
MHD::runCL_EnqueueReadBuffer()
{

    cl_int status_in_runCL;
    cl_event events[2];
//	ctx_device.u --> ctx_host.u
    status_in_runCL = clEnqueueReadBuffer(
		mhdCommandQueue,
                ctx_device.u,
                CL_TRUE,
                0,
                total_cell_number*nu*sizeof(data_type_t),
                ctx_host.u,
                0,
                NULL,
                &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
	printf("Error: clEnqueueReadBuffer in runCL_EnqueueReadBuffer! (ctx_device.u)\n");
	return MHD_FAILURE;
    }
//	ctx_device.b --> ctx_host.b
    status_in_runCL = clEnqueueReadBuffer(
                mhdCommandQueue,
                ctx_device.b,
                CL_TRUE,
                0,
                total_cell_number*nb*sizeof(data_type_t),
                ctx_host.b,
                0,
                NULL,
                &events[1]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueReadBuffer in runCL_EnqueueReadBuffer! (ctx_device.b)\n");
        return MHD_FAILURE;
    }
//
    /* Wait for the read buffer to finish execution */
    status_in_runCL = clWaitForEvents(2, &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clWaitForEvents in runCL_EnqueueReadBuffer!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
//
	return MHD_SUCCESS;
}

int
MHD::runCL_EnqueueCopyBuffer()
{
    cl_int status_in_runCL;
    cl_event events[2];
//      ctx_device.u_update --> ctx_device.u
    status_in_runCL = clEnqueueCopyBuffer(
		mhdCommandQueue,
		ctx_device.u_update,
		ctx_device.u,
		0,
		0,
		total_cell_number*nu*sizeof(data_type_t),
		0,
		NULL,
		&events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueCopyBuffer in runCL_EnqueueCopyBuffer! (ctx_device.u)\n");
        return MHD_FAILURE;
    }
//      ctx_device.b_update --> ctx_device.b
    status_in_runCL = clEnqueueCopyBuffer(
                mhdCommandQueue,
                ctx_device.b_update,
                ctx_device.b,
                0,
                0,
                total_cell_number*nb*sizeof(data_type_t),
                0,
                NULL,
                &events[1]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueCopyBuffer in runCL_EnqueueCopyBuffer! (ctx_device.b)\n");
        return MHD_FAILURE;
    }
//
    /* Wait for the copy buffer to finish execution */
    status_in_runCL = clWaitForEvents(2, &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clWaitForEvents in runCL_EnqueueCopyBuffer!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
//
	return MHD_SUCCESS;
}

int
MHD::runCL_TimingStart()
{
    cl_int   status_in_runCL = CL_SUCCESS;
    cl_event events_TimingStart[1];

    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                timing_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                &events_TimingStart[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on runCL_TimingStart!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on runCL_TimingStart!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clGetEventProfilingInfo(
		events_TimingStart[0],
                CL_PROFILING_COMMAND_END,
                sizeof(long long),
                &startTime,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clGetEventProfilingInfo on runCL_TimingStart!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events_TimingStart[0]);

        return MHD_SUCCESS;
}

int
MHD::runCL_TimingEnd()
{
    cl_int   status_in_runCL = CL_SUCCESS;
    cl_event events_TimingEnd[1];

    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                timing_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                &events_TimingEnd[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on runCL_TimingEnd!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on runCL_TimingEnd!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clGetEventProfilingInfo(
                events_TimingEnd[0],
                CL_PROFILING_COMMAND_START,
                sizeof(long long),
                &endTime,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clGetEventProfilingInfo on runCL_TimingEnd!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events_TimingEnd[0]);

	return MHD_SUCCESS;
}


//	run run run	
int
MHD::runCL_Kernel()
{
//
ct=100.;
tf=ct*1;
//    tf=10.E0;//n*4E0;
    t=0E0;
    iter=0;
do {
if (OutputEveryTimestep)
{
printf("t= %e,  %i,  %e,  %e\n",t,iter,ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(5-1),(5-1),(1-1),(1-1))],ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(1-1),(6-1),(1-1),(1-1))]);
}
    iter=iter+1;
    if (t>=tf) return MHD_SUCCESS;
    if (TimeTheTime_all)    runCL_TimingStart();
//
    if (TimeTheTime_cfl)    runCL_TimingStart();
    runCL_calcfl_Host(); 
    ctx_host.dt=0.9E0*ctx_host.dt;
    ctx_device.dt=ctx_host.dt;
//ctx_device.dt=0.05;
    runCL_UpdateCFLTime();
    if (TimeTheTime_cfl)
    {
        runCL_TimingEnd();
        diffTime = (double)(endTime - startTime)*1.0e-6f;
        printf("CFL openCL time is %f in millisecond\n",diffTime);
    }
//
    if (ctx_device.dt>(tf-t)/2.0E0) ctx_device.dt=(tf-t)/2.0E0;
    t=t+2.0E0*ctx_device.dt;

//	runCL_fluidx_Host();
    //	runCL_advectbyzx_Host();

//      start sweep
    if (TimeTheTime_fluid)    runCL_TimingStart();
    runCL_fluidx_Kernel();
    if (TimeTheTime_fluid)
    {
        runCL_TimingEnd();
        diffTime = (double)(endTime - startTime)*1.0e-6f;
        printf("Fluid openCL time is %f in millisecond\n",diffTime);
    }
    if (TimeTheTime_magnetic)    runCL_TimingStart();
    runCL_advectbyzx_Kernel();
    if (TimeTheTime_magnetic)
    {
        runCL_TimingEnd();
        diffTime = (double)(endTime - startTime)*1.0e-6f;
        printf("Magnetic openCL time is %f in millisecond\n",diffTime);
    }
//      the y sweep
    if (TimeTheTime_transpose)    runCL_TimingStart();
    runCL_transpose12_Kernel();
    if (TimeTheTime_transpose)
    {
        runCL_TimingEnd();
        diffTime = (double)(endTime - startTime)*1.0e-6f;
        printf("Transpose openCL time is %f in millisecond\n",diffTime);
    }

    runCL_fluidx_Kernel();
    runCL_advectbyzx_Kernel();
//	z sweep
    runCL_transpose13_Kernel();
    runCL_fluidx_Kernel();
    runCL_advectbyzx_Kernel();
    runCL_advectbyzx_Kernel();
    runCL_fluidx_Kernel();
//	back
    runCL_transpose13_Kernel();
    runCL_advectbyzx_Kernel();
    runCL_fluidx_Kernel();
//	x again
    runCL_transpose12_Kernel();
    runCL_advectbyzx_Kernel();
    runCL_fluidx_Kernel();

//
    if (TimeTheTime_all)    
    {
	runCL_TimingEnd();
	diffTime = (double)(endTime - startTime)*1.0e-6f;
        diffTime_all=diffTime_all+diffTime;
	printf("All openCL time is %f in millisecond\n",diffTime);
    }
//
if (iter==50)
{
    runCL_CheckValue();
}
    if(runCL_EnqueueReadBuffer() != MHD_SUCCESS)
    {
	printf("Error: runCL_EnqueueReadBuffer in runCL_Kernel()!\n");
	return MHD_FAILURE;
    }
//

    } while (t<tf);
printf("in average, time used in one loop is %f in milisecond\n",diffTime_all/iter);
//
	return MHD_SUCCESS;
}

//	output u and b value to check
int
MHD::runCL_CheckValue()
{
//
int i,j,k;
//
    if(runCL_EnqueueReadBuffer() != MHD_SUCCESS)
    {
        printf("Error: runCL_EnqueueReadBuffer in runCL_Kernel()!\n");
        return MHD_FAILURE;
    }
//
  FILE *check_File_u1;
  FILE *check_File_u2;
  FILE *check_File_u3;
  FILE *check_File_u4;
  FILE *check_File_u5;
  FILE *check_File_b1;
  FILE *check_File_b2;
  FILE *check_File_b3;
//
  check_File_u1=fopen("openCL_check_u1.dat","w");
  check_File_u2=fopen("openCL_check_u2.dat","w");
  check_File_u3=fopen("openCL_check_u3.dat","w");
  check_File_u4=fopen("openCL_check_u4.dat","w");
  check_File_u5=fopen("openCL_check_u5.dat","w");
  check_File_b1=fopen("openCL_check_b1.dat","w");
  check_File_b2=fopen("openCL_check_b2.dat","w");
  check_File_b3=fopen("openCL_check_b3.dat","w");
//
  fclose(check_File_u1);
  fclose(check_File_u2);
  fclose(check_File_u3);
  fclose(check_File_u4);
  fclose(check_File_u5);
  fclose(check_File_b1);
  fclose(check_File_b2);
  fclose(check_File_b3);
//
  check_File_u1=fopen("openCL_check_u1.dat","w+");
  check_File_u2=fopen("openCL_check_u2.dat","w+");
  check_File_u3=fopen("openCL_check_u3.dat","w+");
  check_File_u4=fopen("openCL_check_u4.dat","w+");
  check_File_u5=fopen("openCL_check_u5.dat","w+");
  check_File_b1=fopen("openCL_check_b1.dat","w+");
  check_File_b2=fopen("openCL_check_b2.dat","w+");
  check_File_b3=fopen("openCL_check_b3.dat","w+");
//
        for (k=0;k<(box_nz);k++)
        {
                for (j=0;j<(box_ny);j++)
                {
                        for (i=0;i<(box_nx);i++)
                        {
                                fprintf(check_File_u1,"%8.6E\n",ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(1-1),i,j,k)]);
                                fprintf(check_File_u2,"%8.6E\n",ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(2-1),i,j,k)]);
                                fprintf(check_File_u3,"%8.6E\n",ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(3-1),i,j,k)]);
                                fprintf(check_File_u4,"%8.6E\n",ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(4-1),i,j,k)]);
                                fprintf(check_File_u5,"%8.6E\n",ctx_host.u[a4D_FinC(5,(box_nx),(box_ny),(box_nz),(5-1),i,j,k)]);
                                fprintf(check_File_b1,"%8.6E\n",ctx_host.b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(1-1),i,j,k)]);
                                fprintf(check_File_b2,"%8.6E\n",ctx_host.b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(2-1),i,j,k)]);
                                fprintf(check_File_b3,"%8.6E\n",ctx_host.b[a4D_FinC(3,(box_nx),(box_ny),(box_nz),(3-1),i,j,k)]);
                        }
                }
        }
  fclose(check_File_u1);
  fclose(check_File_u2);
  fclose(check_File_u3);
  fclose(check_File_u4);
  fclose(check_File_u5);
  fclose(check_File_b1);
  fclose(check_File_b2);
  fclose(check_File_b3);
//
	return MHD_SUCCESS;
}


int
MHD::runCL_advectbyzx_Host()
{
    cl_int   status_in_runCL = CL_SUCCESS;
    cl_event events[1];
    runCL_advectbyzx_Kernel();
    status_in_runCL = clEnqueueReadBuffer(
                mhdCommandQueue,
                ctx_device.fluidx_test,
                CL_TRUE,
                0,
                total_cell_number*sizeof(int),
                ctx_host.fluidx_test,
                0,
                NULL,
                &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueReadBuffer in runCL_advectbyzx_GetCFLTime! (ctx_device.fluidx_test)\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clWaitForEvents(1, &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clWaitForEvents in runCL_advectbyzx_GetCFLTime!\n");
        return MHD_FAILURE;
    }
    clReleaseEvent(events[0]);
int i,j,k;
for (k=0;k<(box_nz);k++)
{
        for (j=0;j<(box_ny);j++)
        {
                for (i=0;i<(box_nx);i++)
                {
printf("in advectbyzx: i,j,k,test value: %i,%i,%i,%i\n",i,j,k,ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)]);
                }
        }
}
        return MHD_SUCCESS;

}
int
MHD::runCL_fluidx_Host()
{
    cl_int   status_in_runCL = CL_SUCCESS;
    cl_event events[1];
//	run runCL_fluidx_Kernel
    runCL_fluidx_Kernel();
    status_in_runCL = clEnqueueReadBuffer(
                mhdCommandQueue,
                ctx_device.fluidx_test,
                CL_TRUE,
                0,
                total_cell_number*sizeof(int),
                ctx_host.fluidx_test,
                0,
                NULL,
                &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueReadBuffer in runCL_fluidx_GetCFLTime! (ctx_device.fluidx_test)\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clWaitForEvents(1, &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clWaitForEvents in runCL_fluidx_GetCFLTime!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events[0]);
//
int i,j,k;
for (k=0;k<(box_nz);k++)
{
        for (j=0;j<(box_ny);j++)
        {
		for (i=0;i<(box_nx);i++)
		{		
//printf("in fluidx: i,j,k,test value: %i,%i,%i,%i\n",i,j,k,ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)]);
//if (((ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)])!=32)&&(k>30))
if (((ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)])!=0))
{
//printf("in fluidx: %i,%i,%i,%i\n",i,j,k,ctx_host.fluidx_test[a3D_FinC(box_nx,box_ny,box_nz,i,j,k)]);
//printf("%i\n",i);
//printf("%i\n",i);
//printf("%i\n",k);
}
		}
	}
}
//
        return MHD_SUCCESS;
}

//	calcfl on host
int
MHD::runCL_calcfl_Host()
{

    cl_int   status_in_runCL = CL_SUCCESS;
    cl_event events[1];
//	run runCL_calcfl_Kernel
    runCL_calcfl_Kernel();
//	get the value for each work group
    status_in_runCL = clEnqueueReadBuffer(
                mhdCommandQueue,
                ctx_device.cfl_tmpC,
                CL_TRUE,
                0,
                box_ny*box_nz*sizeof(data_type_t),
                ctx_host.cfl_tmpC,
                0,
                NULL,
                &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clEnqueueReadBuffer in runCL_calcfl_GetCFLTime! (ctx_device.cfl_tmpC)\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clWaitForEvents(1, &events[0]);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: clWaitForEvents in runCL_calcfl_GetCFLTime!\n");
        return MHD_FAILURE;
    }

    clReleaseEvent(events[0]);
//	calculate the max of c
int j,k;
max_c=0.0E0;
for (k=0;k<(box_nz);k++)
{
        for (j=0;j<(box_ny);j++)
        {
                if (ctx_host.cfl_tmpC[a2D_FinC((box_ny),(box_nz),j,k)]>max_c)
                {
                        max_c=ctx_host.cfl_tmpC[a2D_FinC((box_ny),(box_nz),j,k)];
                }
        }
}
ctx_host.dt=1/max_c;
//
	return MHD_SUCCESS;
}

//	for calcfl 
int
MHD::runCL_calcfl_Kernel()
{
    cl_int   status_in_runCL = CL_SUCCESS;

    if(localThreads[0] > mhdMaxWorkItemSizes[0] || localThreads[0] > mhdMaxWorkGroupSize)
    {
        printf("Unsupported: Device does not support requested number of work items!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                calcfl_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on calcfl_Kernel!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on calcfl_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

//	for fluidx function
int
MHD::runCL_fluidx_Kernel()
{
    cl_int   status_in_runCL = CL_SUCCESS;

    if(localThreads[0] > mhdMaxWorkItemSizes[0] || localThreads[0] > mhdMaxWorkGroupSize)
    {
	printf("Unsupported: Device does not support requested number of work items!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clEnqueueNDRangeKernel(
		mhdCommandQueue,
                fluidx_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
	printf("Error: Failed to enqueue a kernel on fluidx_Kernel!\n");
	return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
	printf("Error: Failed to run clFinish on fluidx_Kernel!\n");
        return MHD_FAILURE;
    }

	return MHD_SUCCESS;
}

int
MHD::runCL_advectbyzx_Kernel()
{
    cl_int   status_in_runCL = CL_SUCCESS;

    /*
     *	Enqueue a kernel run call.
     */

    if(localThreads[0] > mhdMaxWorkItemSizes[0] || localThreads[0] > mhdMaxWorkGroupSize)
    {
        printf("Unsupported: Device does not support requested number of work items!\n");
        return MHD_FAILURE;
    }

    /*	clEnqueueNDRangeKernel four kernel
     *	advectbyzxA1_Kernel
     * 	advectbyzxA2_Kernel
     *	advectbyzxB1_Kernel
     *	advectbyzxB2_Kernel
     */	

//	advectbyzxA1_Kernel
    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                advectbyzxA1_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on advectbyzxA1_Kernel!\n");
        return MHD_FAILURE;
    }
    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on advectbyzxA1_Kernel!\n");
        return MHD_FAILURE;
    }


//	advectbyzxA2_Kernel
    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                advectbyzxA2_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on advectbyzxA2_Kernel!\n");
        return MHD_FAILURE;
    }
    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on advectbyzxA2_Kernel!\n");
        return MHD_FAILURE;
    }

//	advectbyzxB1_Kernel
    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                advectbyzxB1_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on advectbyzxB1_Kernel!\n");
        return MHD_FAILURE;
    }
    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on advectbyzxB1_Kernel!\n");
        return MHD_FAILURE;
    }

//	advectbyzxB2_Kernel
    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                advectbyzxB2_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on advectbyzxB2_Kernel!\n");
        return MHD_FAILURE;
    }
    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on advectbyzxB2_Kernel!\n");
        return MHD_FAILURE;
    }
//
        return MHD_SUCCESS;
}

int
MHD::runCL_transpose12_Kernel()
{
    cl_int status_in_runCL = CL_SUCCESS;

    if(localThreads[0] > mhdMaxWorkItemSizes[0] || localThreads[0] > mhdMaxWorkGroupSize)
    {
        printf("Unsupported: Device does not support requested number of work items!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                transpose12_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on transpose12_Kernel!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on transpose12_Kernel!\n");
        return MHD_FAILURE;
    }

//	copy ctx_device.u_update to ctx_device.u
//	copy ctx_device.b_update to ctx_device.b
    if(runCL_EnqueueCopyBuffer() != MHD_SUCCESS)
    {
        printf("Error: runCL_EnqueueCopyBuffer in runCL_transpose12_Kernel()!\n");
        return MHD_FAILURE;
    }

//
        return MHD_SUCCESS;
}

int
MHD::runCL_transpose13_Kernel()
{
    cl_int status_in_runCL = CL_SUCCESS;

    if(localThreads[0] > mhdMaxWorkItemSizes[0] || localThreads[0] > mhdMaxWorkGroupSize)
    {
        printf("Unsupported: Device does not support requested number of work items!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clEnqueueNDRangeKernel(
                mhdCommandQueue,
                transpose13_Kernel,
                2,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                NULL);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue a kernel on transpose13_Kernel!\n");
        return MHD_FAILURE;
    }

    status_in_runCL = clFinish(mhdCommandQueue);
    if (status_in_runCL != CL_SUCCESS)
    {
        printf("Error: Failed to run clFinish on transpose13_Kernel!\n");
        return MHD_FAILURE;
    }
//	copy ctx_device.u_update to ctx_device.u
//      copy ctx_device.b_update to ctx_device.b
    if(runCL_EnqueueCopyBuffer() != MHD_SUCCESS)
    {
        printf("Error: runCL_EnqueueCopyBuffer in runCL_transpose12_Kernel()!\n");
        return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}

int MHD::run()
{
//
    if(runCL_Kernel() != MHD_SUCCESS)
    {
	printf("Error: runCL_Kernel in run()!\n");
	return MHD_FAILURE;
    }
//
	return MHD_SUCCESS;
}
