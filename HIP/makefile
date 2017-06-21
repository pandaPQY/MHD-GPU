exec = Cudamhd

# Define Fortran compiler
CC = /opt/rocm/bin/hipcc -g -pthread
LDPATH =  -L/opt/rocm/hcc/lib 
#LDLIBS =
INC = -I/opt/rocm/hip/include/hip/hcc_detail
FFLAGS = -std=gnu89 
#
cuda_C = /opt/rocm/hip/bin/hipify
cuda_FFLAGS = 
cuda_LDLIBS = -L/opt/rocm/hip/include/hip/ -lhip_hcc
#FC= g95

objects =mhd.o \
         init.o\
         cuda_main.o	\
	 cuda_cfl.o	\
	 cuda_fluidx.o	\
	 cuda_advectbyzx.o	\
	 cuda_transpose12.o	\
	 cuda_transpose13.o
$(exec): $(objects)
	$(CC) $(FFLAGS) $(INC) $(objects) -o $@ $(LDPATH) $(LDLIBS) 
mhd.o: mhd.cpp
	gcc -g $(LDPATH) $(LDLIBS) $(INL) -c mhd.cpp -o mhd.o
init.o: init.cpp
	gcc -g $(LDPATH) $(LDLIBS) $(INL) -c init.cpp -o init.o
cuda_main.o: cuda_main.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_main.cpp -o cuda_main.o -lm
cuda_cfl.o: cuda_cfl.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_cfl.cpp -o cuda_cfl.o -lm
cuda_fluidx.o: cuda_fluidx.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_fluidx.cpp -o cuda_fluidx.o -lm
cuda_advectbyzx.o: cuda_advectbyzx.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_advectbyzx.cpp -o cuda_advectbyzx.o -lm
cuda_transpose12.o: cuda_transpose12.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_transpose12.cpp -o cuda_transpose12.o -lm
cuda_transpose13.o: cuda_transpose13.cpp
	$(CC)  $(FFLAGS) $(LDPATH) $(LDLIBS) $(INC)  -c cuda_transpose13.cpp -o cuda_transpose13.o -lm

#cuda_main.cpp: cuda_main.cu 
#	$(cuda_C) cuda_main.cu > cuda_main.cpp
#cuda_cfl.cpp: cuda_cfl.cu
#	$(cuda_C) cuda_cfl.cu > cuda_cfl.cpp
#cuda_fluidx.cpp: cuda_fluidx.cu
#	$(cuda_C) cuda_fluidx.cu > cuda_fluidx.cpp
#cuda_advectbyzx.cpp: cuda_advectbyzx.cu
#	$(cuda_C) cuda_advectbyzx.cu > cuda_advectbyzx.cpp
#cuda_transpose12.cpp: cuda_transpose12.cu
#	$(cuda_C) cuda_transpose12.cu > cuda_transpose12.cpp
#cuda_transpose13.cpp: cuda_transpose13.cu
#	$(cuda_C) cuda_transpose13.cu > cuda_transpose13.cpp









#hip_function.h: cuda_function.h
#	$(cuda_C) cuda_function.h > hip_function.h
#hip_subroutine.h: cuda_subroutine.h
#	$(cuda_C) cuda_subroutine.h > hip_subroutine.h
#hip_funclist.h: cuda_funclist.h
#	$(cuda_C) cuda_funclist.h > hip_funclist.h

#mhd.o: mhd.c
#	$(CC) $(FFLAGS) -c $< -o $@
#
#init.o: init.c
#	$(CC) $(FFLAGS) -c $< -o $@
#
#cuda_main.o: cuda_main.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
#cuda_cfl.o: cuda_cfl.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
#cuda_fluidx.o: cuda_fluidx.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
#cuda_advectbyzx.o: cuda_advectbyzx.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
#cuda_transpose12.o: cuda_transpose12.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
#cuda_transpose13.o: cuda_transpose13.cu
#	$(cuda_C) $(cuda_FFLAGS) -c $< -o $@ $(cuda_LDLIBS)
#
.PHONY: clean
clean: 
	rm -f $(exec) *.o *.linkinfo

backup:
	tar -cvf backup_cuda_mhd.tar *.c *.cu *.h makefile ; cp backup_cuda_mhd.tar ~/cuda_MHD/BACKUP/
# tar -xvf
