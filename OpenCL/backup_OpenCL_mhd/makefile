machine_selection=orange
#machine_selection=ant
#machine_selection=muskrat
#machine_selection=gamma 		# it doesn't work yet
#
#
exec = mhd 
#
cl_CC = g++
cl_CFLAGS = -O3
cl_LDLIBS = -lOpenCL# -lSDKUtil

ifeq ($(machine_selection),orange)
cl_LDPATH = -L/home/qiaoyin/AMDAPPSDK-2.9-1/lib/x86_64
cl_LDINC = -I/home/qiaoyin/AMDAPPSDK-2.9-1/include
endif

ifeq ($(machine_selection),ant)
cl_LDPATH =  -L/usr/ati-stream-sdk-v2.0-lnx64/lib/x86_64
#cl_LDPATH =  -L/usr/ati-stream-sdk-v2.0-beta4-lnx64/lib/x86_64
cl_LDINC = -I/usr/ati-stream-sdk-v2.0-lnx64/include 
#cl_LDINC = -I/usr/ati-stream-sdk-v2.0-beta4-lnx64/include 
endif

ifeq ($(machine_selection),muskrat)
cl_LDPATH =  -L/usr/lib64/ -L/cita/h/home-1/bpang/NVIDIA_GPU_Computing_SDK/lib
cl_LDINC = -I/cita/h/home-1/bpang/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc 
endif

ifeq ($(machine_selection),gamma)
cl_LDPATH =  -L/cita/h/home-1/bpang/gamma_opencl_lib
cl_LDINC = -I/cita/h/home-1/bpang/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc
endif

objects = mhd.o \
	mhd_setup.o mhd_setup_calcfl.o mhd_setup_fluidx.o mhd_setup_advectbyzx.o \
	mhd_setup_transpose12.o mhd_setup_transpose13.o \
	mhd_run.o mhd_cleanup.o mhd_setup_timing.o

$(exec): $(objects)
	$(cl_CC) $(cl_CFLAGS) $(objects) -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd.o: mhd.cpp 
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup.o: mhd_setup.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_calcfl.o: mhd_setup_calcfl.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_fluidx.o: mhd_setup_fluidx.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_transpose12.o: mhd_setup_transpose12.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_transpose13.o: mhd_setup_transpose13.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_advectbyzx.o: mhd_setup_advectbyzx.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_run.o: mhd_run.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_cleanup.o: mhd_cleanup.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

mhd_setup_timing.o: mhd_setup_timing.cpp
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

.PHONY: clean
clean:
	rm -f $(exec) *.o

backup:
	rm backup_OpenCL_mhd.tar; tar -cvf backup_OpenCL_mhd.tar *.cpp *.hpp *.cl makefile; cp backup_OpenCL_mhd.tar ~/opencl_MHD/BACKUP/; cp backup_OpenCL_mhd.tar ~/BACKUPFORBOB/Heterogeneous_MHD/;

