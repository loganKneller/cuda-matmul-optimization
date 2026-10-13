# Builds the MLP inference benchmark and the SGEMM microbenchmark.
# Override the GPU architecture if nvcc is too old for "native", e.g. make ARCH=sm_75 (T4).
NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: mlp_benchmark gemm

# Built at the repo root: it reads weights/ relative to the working directory.
mlp_benchmark: mlp/mlp_benchmark.cu gemm/kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gemm:
	$(MAKE) -C gemm

clean:
	rm -f mlp_benchmark
	$(MAKE) -C gemm clean

.PHONY: all gemm clean
