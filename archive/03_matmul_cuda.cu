#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

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

// basic matrix multiplication, used to compare to gpu matmul
void cpuMatmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }

            C[i*N + j] = sum;
        }
    }
}

int main() {
    // Using example matrices; they can be arbitrarily large
    const int M = 55;
    const int K = 66;
    const int N = 77;

    unsigned int seed = 100;
    mt19937 gen(seed);
    uniform_real_distribution<float> distrib(-1.0f, 1.0f);

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

    // copy result from GPU back to CPU
    checkCuda(cudaMemcpy(
        h_C, d_C, bytesC, cudaMemcpyDeviceToHost
    ));

    // create and compute an array to store the CPU reference result
    std::vector<float> expected(M * N);
    cpuMatmul(h_A, h_B, expected.data(), M, K, N);

    // now, calculate the error in our GPU-calculated matrix
    const float atol = 1e-3f;
    const float rtol = 1e-4f;

    float maxError = 0.0f;
    int failures = 0;

    for (int i = 0; i < M * N; i++) {
        float gpuValue = h_C[i];
        float cpuValue = expected[i];

        float error = fabs(gpuValue - cpuValue);

        float tolerance = atol + rtol * fabs(cpuValue);

        // track the largest absolute error
        if (std::isfinite(error)) {
            maxError = max(maxError, error);
        }

        // check for invalid numbers or excessive error
        if (!std::isfinite(gpuValue) ||
            !std::isfinite(cpuValue) ||
            error > tolerance) {

            failures++;

            // Print only the first few failures.
            if (failures <= 5) {
                std::cout << "Mismatch at ("
                          << i / N << ", " << i % N << "): "
                          << "CPU = " << cpuValue
                          << ", GPU = " << gpuValue
                          << std::endl;
            }
        }
    }

    std::cout << "Maximum absolute error: "
              << maxError << std::endl;

    if (failures == 0) {
        std::cout << "PASS: All values match within tolerance."
                  << std::endl;
    } else {
        std::cout << "FAIL: " << failures
                  << " values did not match."
                  << std::endl;
    }

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return failures == 0 ? 0: 1;
}
