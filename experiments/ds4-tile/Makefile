# ds4-tile MoE prefill A/B harness
# GB10 / sm_121a (DGX Spark). Falls back at build time is manual: override ARCH.
ARCH   ?= sm_121a
NVCC   ?= nvcc
GGML   ?= ../../ggml/src
NVFLAGS = -O3 -arch=$(ARCH) -lineinfo -I$(GGML) -Xcompiler -fopenmp

all: ds4-tile-harness

ds4-tile-harness: harness.cu ds4_tile_kernels.cuh ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVFLAGS) harness.cu -o $@

clean:
	rm -f ds4-tile-harness

.PHONY: all clean

fp4-oracle: fp4_oracle.cpp
	g++ -O2 -march=native -fopenmp -o $@ fp4_oracle.cpp

