machine_selection=orange
#machine_selection=ATI
#machine_selection=nVidia
#
exec = mhd 
#
cl_CC = gcc
cl_CFLAGS = -O3
cl_LDLIBS = -lOpenCL

ifeq ($(machine_selection),orange)
cl_LDPATH = -L/home/qiaoyin/AMDAPPSDK-2.9-1/lib/x86_64
cl_LDINC = -I/home/qiaoyin/AMDAPPSDK-2.9-1/include
endif

ifeq ($(machine_selection),ATI)
#	on fc10 beta4
cl_LDPATH =  -L/usr/ati-stream-sdk-v2.0-beta4-lnx64/lib/x86_64
cl_LDINC = -I/usr/ati-stream-sdk-v2.0-beta4-lnx64/include 

#	on centOS not beta
#cl_LDPATH =  -L/usr/ati-stream-sdk-v2.0-lnx64/lib/x86_64
#cl_LDINC = -I/usr/ati-stream-sdk-v2.0-lnx64/include 

#	on fc10 not beta
#cl_LDPATH =  -L/mnt/centos/usr/ati-stream-sdk-v2.0-lnx64/lib/x86_64
#cl_LDINC =  -L/mnt/centos/usr/ati-stream-sdk-v2.0-lnx64/  
endif

ifeq ($(machine_selection),nVidia)
cl_LDPATH =  -L/usr/lib64/ -L/cita/h/home-1/bpang/NVIDIA_GPU_Computing_SDK/lib
cl_LDINC = -I/cita/h/home-1/bpang/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc 
endif

objects = mhd.o \
	mhd_setup.o mhd_setup_calcfl.o mhd_setup_fluidx.o mhd_setup_advectbyzx.o \
	mhd_setup_transpose12.o mhd_setup_transpose13.o \
	mhd_run.o mhd_cleanup.o mhd_setup_timing.o

$(exec): $(objects)
	$(cl_CC) $(cl_CFLAGS) $(objects) -o $@ $(cl_LDPATH) $(cl_LDLIBS) $(cl_LDINC)

%.o : %.c
	$(cl_CC) $(cl_CFLAGS) -c $< -o $@ $(cl_LDPATH) $(cl_LDINC)

.PHONY: clean
clean:
	rm -f $(exec) *.o

backup:
	rm backup_C_OpenCL_mhd.tar; tar -cvf backup_C_OpenCL_mhd.tar *.c *.h *.cl makefile; cp backup_C_OpenCL_mhd.tar ~/opencl_MHD/BACKUP/; cp backup_C_OpenCL_mhd.tar ~/BACKUPFORBOB/Heterogeneous_MHD/;

