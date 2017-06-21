FC = ifort

FFLAG = -O3 -qopenmp -qopenmp-offload -mkl -align array64byte#-p# -mkl #-qopt-report=3 -qopt-report-phase=vec
#FFLAG = -qopenmp -qno-offload

all: output

output: work.o mhd.o
	$(FC) $(FFLAG) -o $@ $^

work.o: work.f90
	$(FC) $(FFLAG) -c $<

mhd.o: mhd.f90
	$(FC) $(FFLAG) -c $<
clean:
	rm mhd.o work.o *.mod *.modmic

