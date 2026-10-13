#include <cuda_runtime.h>
#include <iostream>

__global__ void add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 256;
    const size_t bytes = n * sizeof(float);

    float *a, *b, *c;

    // allocate memory accessible to both cpu and gpu
    if (cudaMallocManaged(&a, bytes) != cudaSuccess ||
        cudaMallocManaged(&b, bytes) != cudaSuccess ||
        cudaMallocManaged(&c, bytes) != cudaSuccess) {
        std::cerr << "Allocation failed\n";
        return 1;
    }

    // set up our inputs on the CPU
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 10;
    }

    // launching 1 block of 256 gpu threads
    add<<<1, 256>>>(a, b, c, n);


    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        return 1;
    }


    for (int i = 0; i < n; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << "\n";

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
