#ifndef ARRAY_DEFINITION_H_
#define ARRAY_DEFINITION_H_

//	fortran in C array
#define a4D_FinC(nub,nx,ny,nz,ii,i,j,k) (((((k)*(ny))+(j))*(nx)+(i))*(nub)+(ii))
#define a3D_FinC(nx,ny,nz,i,j,k) ((((k)*(ny))+(j))*(nx)+(i))
#define a2D_FinC(nub,nx,ii,i) (((i)*(nub))+(ii)) 

//	C in C array
#define a4D_CinC(nub,nx,ny,nz,ii,i,j,k) (((((ii)*(nx))+(i))*(ny)+(j))*(nz)+(k))
#define a3D_CinC(nx,ny,nz,i,j,k) ((((i)*(ny))+(j))*(nz)+(k))
#define a2D_CinC(nub,nx,ii,i) (((ii)*(nx))+(i))

#endif	// ARRAY_DEFINITION_H_

