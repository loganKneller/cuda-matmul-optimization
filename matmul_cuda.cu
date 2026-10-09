#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

using namespace std;

// check whether the cuda operation succeeded.
// we wrap all cuda operations in this to handle failures at any point
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    // giving the thread its dimensions to execute
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;


    if (i < M && j < N) {
        float sum = 0;

        // computing dot product
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }

        C[i*N + j] = sum;
    }
}

int main() {
    // Using example matrices; they can be arbitrarily large
    const int M = 256;
    const int K = 256;
    const int N = 256;

    unsigned int seed = 100;
    mt19937 gen(seed);
    uniform_int_distribution<int> distrib(1, 100);
    
    float h_A[M*K];
    float h_B[K*N];
    float h_C[M*N];

    for (int i = 0; i < M; i++){
      for (int j = 0; j < K; j++){
        h_A[i * K + j] = distrib(gen);
      }
    }

    for (int i = 0; i < K; i++){
      for (int j = 0; j < N; j++){
        h_B[i * N + j] = distrib(gen);
      }
    }
    

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    checkCuda(cudaMalloc((void**)&d_A, bytesA));
    checkCuda(cudaMalloc((void**)&d_B, bytesB));
    checkCuda(cudaMalloc((void**)&d_C, bytesC));

    checkCuda(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // initializing block of threads
    // most efficient size depends on the size of input matrix
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    // executes on the GPU
    matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Copy result from GPU back to CPU.
    checkCuda(cudaMemcpy(
        h_C, d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return 0;
}
