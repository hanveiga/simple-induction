CF=gfortran
CCUDA=nvcc
FFLAGS=-lgfortran
#CUDAFLAGS=-lstdc++ -L/usr/local/cuda-7.0/lib64/ -lcuda -lcudart
CUDAFLAGS=-lstdc++ -L/sw/arcts/centos7/cuda/11.0.2/lib64 -lcudart

FOBJS=bin/parameters_dg_2d.o bin/legendre.o bin/limiters.o bin/InitCond.o bin/NodesModes.o bin/Outputs.o bin/source.o bin/Speeds.o bin/update.o bin/fluxes.o bin/utils.o bin/divergence.o
CUDAOBJS=bin/Cuda_functions.o
#CUDAOBJS=bin/Cuda_functions_ldf.o

OUTDIR = bin/
SRCDIR = src/

exe: $(FOBJS) $(CUDAOBJS) $(SRCDIR)main.f90
	$(CF) -o dg2d $(SRCDIR)main.f90 $(FOBJS) $(CUDAOBJS) $(FFLAGS) $(CUDAFLAGS)
$(OUTDIR)%.o $(OUTDIR)%.mod: $(SRCDIR)%.f90
	$(CF) $(FFLAGS) -c $< -o $@
$(OUTDIR)%.o: $(SRCDIR)%.cu
	$(CCUDA) $(CUDAFLAGS) -c $< -o $@
clean:
	rm -f src/*.f90~
	rm -f bin/*.o
	rm -f *.mod
	rm dg2d
