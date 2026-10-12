# Override the GPU architecture if nvcc is too old for "native", e.g. make ARCH=sm_75 (T4).
NVCC ?= nvcc
ARCH ?= native
# -lineinfo maps profiler results back to source lines in Nsight Compute.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

gemm_bench: gemm_bench.cu kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ gemm_bench.cu -lcublas

run: gemm_bench
	./gemm_bench

clean:
	rm -f gemm_bench

.PHONY: run clean
