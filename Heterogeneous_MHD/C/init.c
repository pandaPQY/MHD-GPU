#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "parameter.h"
#include "array_definition.h"
#include "includefunc_ppu.h"

float variable_u[nu*TOTAL_CELL_NUMBER];
float variable_b[nb*TOTAL_CELL_NUMBER];
void init(float *init_u, float *init_b, int *init_nx, int *init_ny, int *init_nz, int *init_total_cell_number, int *init_no_meaning)
{

printf("beginning in init\n");

  *init_nx = box_nx;
  *init_ny = box_ny;
  *init_nz = box_nz;
  *init_total_cell_number = TOTAL_CELL_NUMBER;
  *init_no_meaning = 0;

  int i,j,k,ii;

//	just read the source_init.dat for the data

  FILE *init_File;
  //init_File=fopen("source_init_alfvenlinear_3cube.dat","r+");
  //init_File=fopen("source_init_alfvenlinear_10cube.dat","r+");
  init_File=fopen("source_init_alfvenlinear_16cube.dat","r+");
  //init_File=fopen("source_init_cpaw.dat","r+");
  //init_File=fopen("source_init_magnetic.dat","r+");
  //init_File=fopen("source_init_magnetosonic.dat","r+");
  for (i=0;i<(*init_nx);i++)
  {
	for (j=0;j<(*init_ny);j++)
	{
		for (k=0;k<(*init_nz);k++)
		{
//                        for (ii=0;ii<5;ii++)
 //                       {
 //                                fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),ii,i,j,k)],sizeof(float),1,init_File);
 //                       }
 //                       for (ii=0;ii<3;ii++) 
 //                       {
 //                                fread(&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),ii,i,j,k)],sizeof(float),1,init_File);
 //                       }
                        fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),(1-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),(2-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),(3-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),(4-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),(5-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),(1-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),(2-1),i,j,k)],sizeof(float),1,init_File);
                        fread(&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),(3-1),i,j,k)],sizeof(float),1,init_File);
//			fscanf(init_File,"%f",&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),0,i,j,k)]);
//			fscanf(init_File,"%f",&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),1,i,j,k)]);
//			fscanf(init_File,"%f",&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),2,i,j,k)]);
//			fscanf(init_File,"%f",&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),3,i,j,k)]);
//			fscanf(init_File,"%f",&init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),4,i,j,k)]);
//			fscanf(init_File,"%f",&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),0,i,j,k)]);
//			fscanf(init_File,"%f",&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),1,i,j,k)]);
//			fscanf(init_File,"%f",&init_b[matrix4D(3,(*init_nx),(*init_ny),(*init_nz),2,i,j,k)]);
//			//printf("%d,%d,%d,%f\n",i,j,k,init_u[matrix4D(5,(*init_nx),(*init_ny),(*init_nz),0,i,j,k)]);
		}
	}
  }
  fclose(init_File);

//	finished
printf("end in init\n");
return;

//
//	alfven wave
//	background : rho=B_x=1 all others zero
//	\dot{v_y}+\div_x(-B_y)=0
//	\dot{B_y}+\div_x (       -   v_y)     = 0
//	let v_y=\epsilon sin(2 \pi (x-t)/L)
//	then B_y=-v_y
/*
  define_value_matrix4D(init_u,5,init_nx,init_ny,init_nz,0);
  define_value_matrix4D(init_u,3,init_nx,init_ny,init_nz,0);
  define_value_matrix3D(init_b,3,init_nx,init_ny,init_nz,1,1-1);
  define_value_matrix3D(init_u,5,init_nx,init_ny,init_nz,1,1-1);
  define_value_matrix3D(init_u,5,init_nx,init_ny,init_nz,0.001,5-1);
  float tmp1;
  //double tmp1;
  for (k=0;k<(*init_nz);k++)
  {
        for (j=0;j<(*init_ny);j++)
        {
  		for (i=0;i<(*init_nx);i++)
                {
			tmp1=2*3.14159*((float)(i+1))/((float)(*init_nx));
			//tmp1=2*3.14159*((double)(i+1))/((double)(*init_nx));
			init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(3-1),i,j,k)]=0.1*sin(tmp1);
		}
	}
  }

//  define_value_M2M_matrix4D(init_b,init_u,3,5,2-1,3-1,init_nx,init_ny,init_nz,-1);

float temp[(*init_nx)*(*init_ny)*(*init_nz)];
int imm,imp;
  for (k=0;k<(*init_nz);k++)
  {
        for (j=0;j<(*init_ny);j++)
        {
                for (i=0;i<(*init_nx);i++)
                {
			imm=(i+(*init_nx)-1)%(*init_nx);
			imp=(i+1)%(*init_nx);
			temp[matrix3D(*init_nx,*init_ny,*init_nz,i,j,k)]=init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(2-1),imm,j,k)];
                }
        }
  }
  for (k=0;k<(*init_nz);k++)
  {
        for (j=0;j<(*init_ny);j++)
        {
                for (i=0;i<(*init_nx);i++)
                {
			init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(2-1),i,j,k)]=(init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(2-1),i,j,k)]+temp[matrix3D(*init_nx,*init_ny,*init_nz,i,j,k)])/2;
                }
        }
  }
  for (k=0;k<(*init_nz);k++)
  {
        for (j=0;j<(*init_ny);j++)
        {
                for (i=0;i<(*init_nx);i++)
                {
			init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(5-1),i,j,k)]=init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(5-1),i,j,k)]+(init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(1-1),i,j,k)]*init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(1-1),i,j,k)]+init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(2-1),i,j,k)]*init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(2-1),i,j,k)]+init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(3-1),i,j,k)]*init_b[matrix4D(3,*init_nx,*init_ny,*init_nz,(3-1),i,j,k)])/2+init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(3-1),i,j,k)]*init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(3-1),i,j,k)]/init_u[matrix4D(5,*init_nx,*init_ny,*init_nz,(1-1),i,j,k)]/2;
                }
        }
  }
//
  return;
*/
}
