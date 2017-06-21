#ifndef MHD_H_
#define MHD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#include "parameter.h"
#include "struct_def.h"

//using std::string;

//	device type
    cl_device_type mhdDeviceType;       
//	compute context
    cl_context mhdComputeContext;       
//	compute device id
    cl_device_id *mhdComputeDevice;     
//	compute command queue
    cl_command_queue mhdCommandQueue;   
//	property for command queue
    cl_command_queue_properties mhdProp;
//	calcfl program and kernel
    cl_program calcfl_Program;          
    cl_kernel calcfl_Kernel;            
//	fluidx program and kernel 
    cl_program fluidx_Program; 
    cl_kernel fluidx_Kernel;   
//	advectbyzx program and kernel
    cl_program advectbyzx_Program;      
    cl_kernel advectbyzx_Kernel;        
    cl_program advectbyzxA1_Program;    
    cl_kernel advectbyzxA1_Kernel;      
    cl_program advectbyzxA2_Program;    
    cl_kernel advectbyzxA2_Kernel;      
    cl_program advectbyzxB1_Program;    
    cl_kernel advectbyzxB1_Kernel;      
    cl_program advectbyzxB2_Program;    
    cl_kernel advectbyzxB2_Kernel;      
//	transpose program and kernel
    cl_program transpose12u1234_Program;     
    cl_kernel transpose12u1234_Kernel;       
    cl_program transpose12u5b_Program;
    cl_kernel transpose12u5b_Kernel;
    cl_program transpose13u1234_Program;     
    cl_kernel transpose13u1234_Kernel;       
    cl_program transpose13u5b_Program;
    cl_kernel transpose13u5b_Kernel;
//	timing program and kernel
    cl_program timing_Program;          
    cl_kernel timing_Kernel;            
//	Max allowed work-items in a group
    size_t mhdMaxWorkGroupSize;         
//	Max group dimensions allowed
    cl_uint mhdMaxDimensions;           
//	Max work-items sizes in each dimensions
    size_t *mhdMaxWorkItemSizes;        
//	Max local memory allowed
    cl_ulong mhdTotalLocalMemory;       
//	Used local memory
    cl_ulong mhdUsedLocalMemory;        
//	Kernel work group size
    size_t mhdKernelWorkGroupSize;      
//	simulation related
    int WorkGroupSize;
    float t;
    float ct;
    float tf;
    int iter;
    float max_c;
    float CFL_value;
//	time measurement
    long long startTime;
    long long endTime;
    double diffTime;
//	variable on host
    variable_context_host ctx_host;
//	variable on device
    variable_context_device ctx_device;

    /**
     * initialize the value of matrix u and b
     */
    void init_value(cl_float *initV_u1234, float *initV_u5, cl_float *initV_b, int *initV_nx, int *initV_ny, int *initV_nz, float *initV_adv_tmpB, cl_float *initV_u_update1234, float *initV_u_update5, cl_float *initV_b_update);

    /**
     * Allocate and initialize value in host memory
     */
    int setupMHD_initialization();

    /**
     * OpenCL related initialisations. 
     * Set up Context, Device list, Command Queue, Memory buffers
     */
    int setupCL_ContextCommandMemory();

    /**
     * Build CL Program and Kernel executable 
     */
    int setupCL_ProgramKernel();
    int setupCL_calcfl_ProgramKernel();
    int setupCL_fluidx_ProgramKernel();
    int setupCL_advectbyzx_ProgramKernel();
    int setupCL_advectbyzxA1_ProgramKernel();
    int setupCL_advectbyzxA2_ProgramKernel();
    int setupCL_advectbyzxB1_ProgramKernel();
    int setupCL_advectbyzxB2_ProgramKernel();
    int setupCL_transpose12u1234_ProgramKernel();
    int setupCL_transpose12u5b_ProgramKernel();
    int setupCL_transpose13u1234_ProgramKernel();
    int setupCL_transpose13u5b_ProgramKernel();
    int setupCL_timing_ProgramKernel();

    /**
     * Set values for kernels' arguments
     */
    int setupCL_ArgumentKernel();
    int setupCL_calcfl_ArgumentKernel();
    int setupCL_fluidx_ArgumentKernel();
    int setupCL_advectbyzx_ArgumentKernel();
    int setupCL_advectbyzxA1_ArgumentKernel();
    int setupCL_advectbyzxA2_ArgumentKernel();
    int setupCL_advectbyzxB1_ArgumentKernel();
    int setupCL_advectbyzxB2_ArgumentKernel();
    int setupCL_transpose12u1234_ArgumentKernel();
    int setupCL_transpose12u5b_ArgumentKernel();
    int setupCL_transpose13u1234_ArgumentKernel();
    int setupCL_transpose13u5b_ArgumentKernel();
    int setupCL_timing_ArgumentKernel();

    /**
     * Enqueue calls to the kernels
     * on to the command queue, wait till end of kernel execution.
     */
    int runCL_Kernel();
    int runCL_calcfl_Kernel();
    int runCL_calcfl_Host();
    int runCL_fluidx_Kernel();
    int runCL_advectbyzx_Kernel();
    int runCL_transpose12_Kernel();
    int runCL_transpose13_Kernel();
    int runCL_fluidx_Host();
    int runCL_advectbyzx_Host();
    /**
     * Read buffer from device to host
     * For test or output
     */
    int runCL_EnqueueReadBuffer();
    int runCL_EnqueueCopyBuffer();
    int runCL_UpdateCFLTime();
    int runCL_CheckValue();
    int runCL_TimingStart();
    int runCL_TimingEnd();

    /**
     * Combine all the setup sub function above
     * of execution domain, perform all sample setup
     */
    int setup();

    /**
     * Combine all the run sub function above 
     * Run OpenCL MHD
     */
    int run();

    /**
     * Cleanup memory allocations
     */
    int cleanup();
    int cleanup_ReleaseKernel();
    int cleanup_calcfl_ReleaseKernel();
    int cleanup_fluidx_ReleaseKernel();
    int cleanup_advectbyzx_ReleaseKernel();
    int cleanup_transpose12u1234_ReleaseKernel();
    int cleanup_transpose12u5b_ReleaseKernel();
    int cleanup_transpose13u1234_ReleaseKernel();
    int cleanup_transpose13u5b_ReleaseKernel();
    int cleanup_timing_ReleaseKernel();
    int cleanup_ReleaseProgram();
    int cleanup_calcfl_ReleaseProgram();
    int cleanup_fluidx_ReleaseProgram();
    int cleanup_advectbyzx_ReleaseProgram();
    int cleanup_transpose12u1234_ReleaseProgram();
    int cleanup_transpose12u5b_ReleaseProgram();
    int cleanup_transpose13u1234_ReleaseProgram();
    int cleanup_transpose13u5b_ReleaseProgram();
    int cleanup_timing_ReleaseProgram();


#endif // MHD_H_ 
