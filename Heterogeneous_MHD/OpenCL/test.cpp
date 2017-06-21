#include<iostream>
#include<stdio.h>
#include<fstream>
#include<math.h>
#include<string>
#ifndef ARRAY_DEFINITION_H_
#define ARRAY_DEFINITION_H_

//      fortran in C array
#define a4D_FinC(nub,nx,ny,nz,ii,i,j,k) (((((k)*(ny))+(j))*(nx)+(i))*(nub)+(ii))
#define a3D_FinC(nx,ny,nz,i,j,k) ((((k)*(ny))+(j))*(nx)+(i))
#define a2D_FinC(nub,nx,ii,i) (((i)*(nub))+(ii)) 

#endif  // ARRAY_DEFINITION_H_


using namespace std;
int main(){
const int n=16;
float u[5*n*n*n];
int i,j,k,ii;
//FILE *fil;
ifstream infile;
infile.open("source_init_alfvenlinear_16cube.dat",ios::binary|ios::in);
//fil=fopen("source_init_alfvenlinear_16cube.dat","r");
for (k=0;k<n;k++){
    for (j=0;j<n;j++){
        for (i=0;i<n;i++){
            for (ii=0;ii<5;ii++){
//                fscanf(fil,"%f",&u[a4D_FinC(5,n,n,n,ii,i,j,k)]);
//                printf("u, %f \n",u[a4D_FinC(5,n,n,n,ii,i,j,k)]);
                infile.read((char*)&u[a4D_FinC(5,n,n,n,ii,i,j,k)],sizeof(float));
                cout<<u[a4D_FinC(5,n,n,n,ii,i,j,k)]<<"\n";
                }
            }
       }
    }
cout<<"u is "<<*u<<"\n";
infile.close();
return 0;
}
