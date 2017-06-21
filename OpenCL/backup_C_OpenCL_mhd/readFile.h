#ifndef READFILE_H_
#define READFILE_H_

void readFile()
{
#include <stdio.h>

const char *ptr;
  FILE *init_File;
  init_File=fopen("source_init_alfvenlinear_128cube.dat","r+");

//long size = filelength(fileno(init_File));
//printf("%d;\n",lseek(init_File,0,SEEK_END));
fseek(init_File, 0, SEEK_END);
int size = ftell(init_File);
fseek(init_File, 0, SEEK_SET);
fread((void *)ptr,sizeof(char),size,init_File);
printf("*ptr is %d\n",*(ptr+20));
printf("size is %d\n",size);
//fseek(init_File,0L,2);
}
#endif
