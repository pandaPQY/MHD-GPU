#ifndef STRUCT_DEF_H_
#define STRUCT_DEF_H_

#include "data_type.h"

//	struct in host
typedef struct{
  cl_float *u1234;	// fluid component, u1, u2, u3, u4
  cl_float *u5;		// fluid component, u5
  cl_float *b;		// magnetic component, b1, b2, b3
  int nx;		// X length
  int ny;		// Y length
  int nz;		// Z length
  cl_float dt;		// cfl time
  cl_float *adv_tmpB;	// matrix for magnetic in magnetic update function
  cl_float *u_update1234;	// matrix for fluid in transpose function, u1, u2, u3, u4
  cl_float *u_update5;	// matrix for fluid in transpose function, u5
  cl_float *b_update;	// matrix for magnetic in transpose function, b1, b2, b3
  cl_float *cfl_tmpC;	// matrix for cfl calculation
  int *fluidx_test;	// 
  cl_float CFL_value;	// cfl value from cfl subroutine
}variable_context_host;

//	struct in device
typedef struct{
  cl_mem u1234;                	// fluid, u1, u2, u3, u4
  cl_mem u5;                  	// fluid, u5
  cl_mem b;                     // magnetic
  cl_int nx;                    // X length
  cl_int ny;                    // Y length
  cl_int nz;                    // Z length
  cl_float dt;                  // cfl time
  cl_mem adv_tmpB;              // matrix for magnetic in magnetic update function
  cl_mem u_update1234;		// matrix for fluid in transpose function, u1, u2, u3, u4
  cl_mem u_update5;             // matrix for fluid in transpose function, u5
  cl_mem b_update;              // matrix for magnetic in transpose function 
  cl_mem cfl_tmpC;		// matrix for cfl calculation
  cl_mem fluidx_test;//
  cl_float CFL_value;		// cfl value from cfl subroutine
}variable_context_device;

#endif	// STRUCT_DEF_H_

