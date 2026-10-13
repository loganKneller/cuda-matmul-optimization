#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

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


__global__ void addBias(
    float* C,
    const float* biases,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M * N) {
        int j = idx % N;
        C[idx] += biases[j];
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
    const int M = 64; //number of images
    const int K = 784; //pixels per image
    const int N = 10; //number of outputs (digits 0-9)

        
    vector<float> h_A(M * K);
    vector<float> h_B(K * N);
    vector<float> h_C(M * N);
    vector<float> biases(N);

    // reading exported MNIST files
    ifstream inputFile("weights/linear/inputs_cuda.bin", ios::binary);
    ifstream weightFile("weights/linear/weights_cuda.bin", ios::binary);
    ifstream biasFile("weights/linear/biases_cuda.bin", ios::binary);

    if (!inputFile || !weightFile || !biasFile) {
        cerr << "Error opening MNIST binary files :(" << endl;
        return EXIT_FAILURE;
    }

    inputFile.read(
        reinterpret_cast<char*>(h_A.data()),
        h_A.size() * sizeof(float)
    );

    weightFile.read(
        reinterpret_cast<char*>(h_B.data()),
        h_B.size() * sizeof(float)
    );

    biasFile.read(
        reinterpret_cast<char*>(biases.data()),
        biases.size() * sizeof(float)
    );

    if (!inputFile || !weightFile || !biasFile) {
        cerr << "Error reading MNIST binary files!" << endl;
        return EXIT_FAILURE;
    }

    cout << "Successfully loaded MNIST data" << endl;
    // at this point, h_A contains the MNIST images, h_B contains the transposed weight matrix

    // now, load pytorch reference output
    vector<float> reference(M * N);

    ifstream referenceFile(
        "weights/linear/reference_logits.bin",
        ios::binary
    );

    if (!referenceFile) {
        cerr << "Could not open PyTorch reference file!" << endl;
        return EXIT_FAILURE;
    }

    referenceFile.read(
        reinterpret_cast<char*>(reference.data()),
        reference.size() * sizeof(float)
    );

    if (!referenceFile) {
        cerr << "Failed to read PyTorch reference!" << endl;
        return EXIT_FAILURE;
    }


    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_biases = nullptr;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t bytesBiases = N * sizeof(float);

    checkCuda(cudaMalloc((void**)&d_A, bytesA));
    checkCuda(cudaMalloc((void**)&d_B, bytesB));
    checkCuda(cudaMalloc((void**)&d_C, bytesC));
    checkCuda(cudaMalloc((void**)&d_biases, bytesBiases));

    checkCuda(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_biases, biases.data(), bytesBiases, cudaMemcpyHostToDevice));

    // initializing block of threads
    // most efficient size depends on the size of input matrix
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    // executes on the GPU
    // calculating XW
    matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    checkCuda(cudaGetLastError());

    // add the learned biases
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    addBias<<<blocks, threads>>>(d_C, d_biases, M, N);
    checkCuda(cudaGetLastError());

    checkCuda(cudaDeviceSynchronize());


    // copy result from GPU back to CPU
    checkCuda(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    // create and compute an array to store the CPU reference result
    std::vector<float> expected(M * N);
    cpuMatmul(h_A.data(), h_B.data(), expected.data(), M, K, N);

    // add the learned biases to CPU output
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            expected[i * N + j] += biases[j];
        }
    }

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


    int torchFailures = 0;
    int matchingPredictions = 0;
    float maxTorchError = 0.0f;

    for (int i = 0; i < M; i++) {
        int gpuPrediction = 0;
        int torchPrediction = 0;

        for (int j = 0; j < N; j++) {
            int idx = i * N + j;

            float gpuValue = h_C[idx];
            float torchValue = reference[idx];

            // compare actual numerical outputs
            float error = fabs(gpuValue - torchValue);
            float tolerance = atol + rtol * fabs(torchValue);

            if (std::isfinite(error)) {
                maxTorchError = max(maxTorchError, error);
            }

            if (!std::isfinite(gpuValue) ||
                !std::isfinite(torchValue) ||
                error > tolerance) {
                torchFailures++;
            }

            // find digit with lagest score
            if (gpuValue > h_C[i * N + gpuPrediction]) {
                gpuPrediction = j;
            }

            if (torchValue > reference[i * N + torchPrediction]) {
                torchPrediction = j;
            }
        }

        if (gpuPrediction == torchPrediction) {
            matchingPredictions++;
        }

        if (i < 10) {
            cout << "Image " << i
                << ": CUDA predicts " << gpuPrediction
                << ", PyTorch predicts " << torchPrediction
                << endl;
        }
    }

    cout << "\nMaximum PyTorch error: "
        << maxTorchError << endl;

    cout << "Logit mismatches: "
        << torchFailures << endl;

    cout << "Matching predictions: "
        << matchingPredictions << "/" << M << endl;


    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    checkCuda(cudaFree(d_biases));

    return (failures == 0 && torchFailures == 0) ? 0: 1;
}
