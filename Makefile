dir = $(shell pwd)

VPATH = $(dir)/utils/ $(dir)/ACSR/ $(dir)/experimental/windowed/

## The path to eigen may be different depending on the machine you are working on. PLEASE CHANGE ACCORDINGLY. ##
CUDAFLAGS = -O3 -I /home/selin/splat-fusion/eigen --expt-relaxed-constexpr -std=c++14

windowed : windowed.o acsr.o utils.o
	nvcc $(CUDAFLAGS) $^ -o windowed_exec

windowed.o : window.cu
	nvcc $(CUDAFLAGS) -c $< -o windowed.o

## Helper recipes. ##
acsr.o : acsr.cc
	nvcc $(CUDAFLAGS) -c $< -o acsr.o

utils.o : utils.cc 
	nvcc $(CUDAFLAGS) -c $< -o utils.o

clean : 
	rm *.o *_exec sddmm_kernel.cu block_placement.cc
