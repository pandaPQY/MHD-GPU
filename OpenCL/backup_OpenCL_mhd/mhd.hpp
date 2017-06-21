#ifndef MHD_H_
#define MHD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#include "parameter.hpp"
#include "struct_def.hpp"

using std::string;

/**
 * MHD 
 * Class implements OpenCL MHD sample
 */

class MHD
{

    cl_device_type mhdDeviceType;       // device type
    cl_context mhdComputeContext;       // compute context
    cl_device_id *mhdComputeDevice;     // compute device id
    cl_command_queue mhdCommandQueue;   // compute command queue
    cl_command_queue_properties mhdProp;// property for command queue 
//	calcfl related
    cl_program calcfl_Program;          // compute calcfl program
    cl_kernel calcfl_Kernel;            // compute calcfl kernel
//	fluidx related
    cl_program fluidx_Program;          // compute fluidx program
    cl_kernel fluidx_Kernel;            // compute fluidx kernel
//	advectbyzx related
    cl_program advectbyzx_Program;      // compute fluidx program
    cl_kernel advectbyzx_Kernel;        // compute fluidx kernel
    cl_program advectbyzxA1_Program;    // compute fluidx program
    cl_kernel advectbyzxA1_Kernel;      // compute fluidx kernel
    cl_program advectbyzxA2_Program;    // compute fluidx program
    cl_kernel advectbyzxA2_Kernel;      // compute fluidx kernel
    cl_program advectbyzxB1_Program;    // compute fluidx program
    cl_kernel advectbyzxB1_Kernel;      // compute fluidx kernel
    cl_program advectbyzxB2_Program;    // compute fluidx program
    cl_kernel advectbyzxB2_Kernel;      // compute fluidx kernel
//	transpose related
    cl_program transpose12_Program;     // compute fluidx program
    cl_kernel transpose12_Kernel;       // compute fluidx kernel
    cl_program transpose13_Program;     // compute fluidx program
    cl_kernel transpose13_Kernel;       // compute fluidx kernel
//	timing related
    cl_program timing_Program;          // compute fluidx program
    cl_kernel timing_Kernel;            // compute fluidx kernel
//
    size_t mhdMaxWorkGroupSize;         // Max allowed work-items in a group
    cl_uint mhdMaxDimensions;           // Max group dimensions allowed
    size_t *mhdMaxWorkItemSizes;        // Max work-items sizes in each dimensions
    cl_ulong mhdTotalLocalMemory;       // Max local memory allowed
    cl_ulong mhdUsedLocalMemory;        // Used local memory
    size_t mhdKernelWorkGroupSize;      // Kernel work group size
//
    int WorkGroupSize;
    data_type_t t;
    data_type_t ct;
    data_type_t tf;
    int iter;
    data_type_t max_c;
//
    long long startTime;
    long long endTime;
    double diffTime;
//
    variable_context_host ctx_host;
    variable_context_device ctx_device;

private:

public:

//    ~MHD();

    /**
     * initialize the value of matrix u and b
     */
    void init_value(data_type_t *initV_u, data_type_t *initV_b, int *initV_nx, int *initV_ny, int *initV_nz, data_type_t *init_adv_tmpB, data_type_t *init_u_update, data_type_t *init_b_update);

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
    int setupCL_transpose12_ProgramKernel();
    int setupCL_transpose13_ProgramKernel();
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
    int setupCL_transpose12_ArgumentKernel();
    int setupCL_transpose13_ArgumentKernel();
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
    int cleanup_transpose12_ReleaseKernel();
    int cleanup_transpose13_ReleaseKernel();
    int cleanup_timing_ReleaseKernel();
    int cleanup_ReleaseProgram();
    int cleanup_calcfl_ReleaseProgram();
    int cleanup_fluidx_ReleaseProgram();
    int cleanup_advectbyzx_ReleaseProgram();
    int cleanup_transpose12_ReleaseProgram();
    int cleanup_transpose13_ReleaseProgram();
    int cleanup_timing_ReleaseProgram();

};

#endif // MHD_H_ 
