#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include "gemm/kernels.cuh"

using namespace std;

// Exit if a CUDA runtime call fails.
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// Matrix multiplication kernels live in gemm/kernels.cuh.
// Keeping them in one header ensures GEMM microbenchmarks and MLP inference
// use the exact same implementations.

// Add the same length-N bias vector to every row of C.
__global__ void addBias(float* C, const float* biases, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < M * N) {
        int j = idx % N;
        C[idx] += biases[j];
    }
}

// In-place ReLU activation.
__global__ void relu(float* values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

bool loadBinary(const string& path, vector<float>& destination) {
    ifstream file(path, ios::binary);

    if (!file) {
        cerr << "Could not open " << path << endl;
        return false;
    }

    file.read(
        reinterpret_cast<char*>(destination.data()),
        destination.size() * sizeof(float)
    );

    if (!file) {
        cerr << "Could not read all values from " << path << endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    // By default, use the fastest kernels measured for each MLP shape.
    // --naive preserves the original baseline without duplicating kernel code.
    bool useNaive = false;
    if (argc == 2 && string(argv[1]) == "--naive") {
        useNaive = true;
    } else if (argc != 1) {
        cerr << "Usage: " << argv[0] << " [--naive]" << endl;
        return EXIT_FAILURE;
    }

    const int M = 10000;       // Number of test images
    const int INPUT = 784;    // Pixels per image
    const int HIDDEN = 128;   // Neurons in hidden layer
    const int OUTPUT = 10;    // Digit scores per image

    // Host-side model inputs, parameters, and outputs.
    vector<float> h_X(M * INPUT);
    vector<float> h_W1(INPUT * HIDDEN);
    vector<float> h_b1(HIDDEN);
    vector<float> h_W2(HIDDEN * OUTPUT);
    vector<float> h_b2(OUTPUT);
    vector<float> h_Z(M * OUTPUT);

    if (
        !loadBinary("weights/mlp/full_test/inputs_cuda.bin", h_X) ||
        !loadBinary("weights/mlp/weights1_cuda.bin", h_W1) ||
        !loadBinary("weights/mlp/biases1_cuda.bin", h_b1) ||
        !loadBinary("weights/mlp/weights2_cuda.bin", h_W2) ||
        !loadBinary("weights/mlp/biases2_cuda.bin", h_b2)
    ) {
        return EXIT_FAILURE;
    }
    cout << "Successfully loaded all MLP data" << endl;

    vector<float> reference(M * OUTPUT);
    if (!loadBinary("weights/mlp/full_test/reference_logits.bin", reference)) {
        return EXIT_FAILURE;
    }
    cout << "Successfully loaded PyTorch reference" << endl;

    // Labels are int32, rather than float.
    vector<int32_t> labels(M);
    ifstream labelFile("weights/mlp/full_test/labels.bin", ios::binary);

    if (!labelFile) {
        cerr << "Could not open MNIST labels" << endl;
        return EXIT_FAILURE;
    }

    labelFile.read(
        reinterpret_cast<char*>(labels.data()),
        labels.size() * sizeof(int32_t)
    );

    if (!labelFile) {
        cerr << "Failed to read MNIST labels" << endl;
        return EXIT_FAILURE;
    }
    cout << "Successfully loaded " << labels.size() << " MNIST labels" << endl;

    // Device-side arrays.
    float* d_X = nullptr;
    float* d_W1 = nullptr;
    float* d_b1 = nullptr;
    float* d_H = nullptr;
    float* d_W2 = nullptr;
    float* d_b2 = nullptr;
    float* d_Z = nullptr;

    checkCuda(cudaMalloc((void**)&d_X, h_X.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_W1, h_W1.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b1, h_b1.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_H, M * HIDDEN * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_W2, h_W2.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b2, h_b2.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_Z, h_Z.size() * sizeof(float)));

    // Transfer inputs and learned parameters once, before timing.
    checkCuda(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    cout << "Successfully copied MLP data to GPU!" << endl;

    // Compute all launch configurations once, outside the timed region.
    // GEMM kernels use row-major matrices, just like the exported PyTorch weights.
    //
    // Layer 1 (10000x784 times 784x128): 2D register blocking, BK=16,
    // with bias and ReLU fused into the GEMM's final write.
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    const dim3 block1(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    const dim3 grid1((HIDDEN + BN - 1) / BN, (M + BM - 1) / BM);

    // Layer 2 (10000x128 times 128x10): simple shared-memory tiling.
    // A 64-column 2D-blocked tile wastes most columns when N is only 10.
    const dim3 block2(TILE, TILE);
    const dim3 grid2((OUTPUT + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Original naive launch configuration, for an apples-to-apples baseline.
    const dim3 naiveBlock(16, 16);
    const dim3 naiveGrid1(
        (HIDDEN + naiveBlock.x - 1) / naiveBlock.x,
        (M + naiveBlock.y - 1) / naiveBlock.y
    );
    const dim3 naiveGrid2(
        (OUTPUT + naiveBlock.x - 1) / naiveBlock.x,
        (M + naiveBlock.y - 1) / naiveBlock.y
    );

    const int threads = 256;
    const int hiddenSize = M * HIDDEN;
    const int hiddenBlocks = (hiddenSize + threads - 1) / threads;
    const int outputSize = M * OUTPUT;
    const int outputBlocks = (outputSize + threads - 1) / threads;

    cout << "Inference mode: " << (useNaive ? "naive GEMM baseline" : "shape-optimized GEMM") << endl;
    if (!useNaive) {
        cout << "  Layer 1: 2D register blocking (BM=64, BN=64, BK=16, TM=4, TN=4) + fused bias/ReLU" << endl;
        cout << "  Layer 2: shared-memory tiling (TILE=" << TILE << ")" << endl;
    }

    // Naive mode launches five kernels; optimized mode fuses the hidden layer's
    // bias and ReLU into its GEMM, leaving three.
    // All kernels execute in the default CUDA stream, in launch order.
    auto runInference = [&]() {
        // Hidden layer: H = ReLU(X * W1 + b1)
        if (useNaive) {
            naiveGemm<<<naiveGrid1, naiveBlock>>>(
                d_X, d_W1, d_H, M, INPUT, HIDDEN
            );
            checkCuda(cudaGetLastError());

            addBias<<<hiddenBlocks, threads>>>(d_H, d_b1, M, HIDDEN);
            checkCuda(cudaGetLastError());

            relu<<<hiddenBlocks, threads>>>(d_H, hiddenSize);
        } else {
            blocktiling2dGemm<BM, BN, BK, TM, TN, true><<<grid1, block1>>>(
                d_X, d_W1, d_H, M, INPUT, HIDDEN, d_b1
            );
        }
        checkCuda(cudaGetLastError());

        // Output layer: Z = H * W2 + b2
        if (useNaive) {
            naiveGemm<<<naiveGrid2, naiveBlock>>>(
                d_H, d_W2, d_Z, M, HIDDEN, OUTPUT
            );
        } else {
            tiledGemm<<<grid2, block2>>>(
                d_H, d_W2, d_Z, M, HIDDEN, OUTPUT
            );
        }
        checkCuda(cudaGetLastError());

        addBias<<<outputBlocks, threads>>>(d_Z, d_b2, M, OUTPUT);
        checkCuda(cudaGetLastError());
    };

    const int warmupRuns = 5;
    const int measuredRuns = 30;

    // Warm up without including these launches in the reported timings.
    for (int i = 0; i < warmupRuns; i++) {
        runInference();
    }
    checkCuda(cudaDeviceSynchronize());

    // Time the entire forward pass, excluding allocations,
    // host/device transfers, data loading, and CPU-side correctness checks.
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    vector<float> times;
    times.reserve(measuredRuns);

    for (int i = 0; i < measuredRuns; i++) {
        checkCuda(cudaEventRecord(start));

        runInference();

        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop));
        times.push_back(elapsedMs);
    }

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    sort(times.begin(), times.end());
    const float medianMs = (times[measuredRuns / 2 - 1] + times[measuredRuns / 2]) / 2.0f;

    cout << fixed << setprecision(4);
    cout << "Warm-up runs: " << warmupRuns << endl;
    cout << "Measured runs: " << measuredRuns << endl;
    cout << "Minimum CUDA inference time: " << times.front() << " ms" << endl;
    cout << "Median CUDA inference time: " << medianMs << " ms" << endl;
    cout << "Maximum CUDA inference time: " << times.back() << " ms" << endl;
    cout << "Amortized median time per image: " << setprecision(7)
         << medianMs / M << " ms" << setprecision(4) << endl;

    // Transfer output after the final measured run (excluded from timing).
    checkCuda(cudaMemcpy(
        h_Z.data(), d_Z, h_Z.size() * sizeof(float), cudaMemcpyDeviceToHost
    ));
    cout << "MLP forward pass completed and output copied to CPU" << endl;

    // Validate the result against PyTorch and the ground-truth MNIST labels.
    const float atol = 1e-3f;
    const float rtol = 1e-4f;
    int torchFailures = 0;
    int matchingPredictions = 0;
    int correctPredictions = 0;
    float maxTorchError = 0.0f;

    for (int i = 0; i < M; i++) {
        int gpuPrediction = 0;
        int torchPrediction = 0;

        for (int j = 0; j < OUTPUT; j++) {
            const int idx = i * OUTPUT + j;
            const float gpuValue = h_Z[idx];
            const float torchValue = reference[idx];
            const float error = fabs(gpuValue - torchValue);
            const float tolerance = atol + rtol * fabs(torchValue);

            if (isfinite(error)) {
                maxTorchError = max(maxTorchError, error);
            }

            if (!isfinite(gpuValue) || !isfinite(torchValue) || error > tolerance) {
                torchFailures++;
            }

            if (gpuValue > h_Z[i * OUTPUT + gpuPrediction]) {
                gpuPrediction = j;
            }
            if (torchValue > reference[i * OUTPUT + torchPrediction]) {
                torchPrediction = j;
            }
        }

        if (gpuPrediction == torchPrediction) {
            matchingPredictions++;
        }
        if (gpuPrediction == labels[i]) {
            correctPredictions++;
        }

        if (i < 10) {
            cout << "Image " << i
                 << ": CUDA predicts " << gpuPrediction
                 << ", PyTorch predicts " << torchPrediction << endl;
        }
    }

    cout << "Maximum PyTorch error: " << scientific << maxTorchError << defaultfloat << endl;
    cout << "Logit mismatches: " << torchFailures << endl;
    cout << "Matching predictions: " << matchingPredictions << "/" << M << endl;
    cout << "Correct predictions: " << correctPredictions << "/" << M << endl;
    cout << "CUDA MNIST accuracy: " << 100.0 * correctPredictions / M << "%" << endl;

    // Release all GPU allocations.
    checkCuda(cudaFree(d_X));
    checkCuda(cudaFree(d_W1));
    checkCuda(cudaFree(d_b1));
    checkCuda(cudaFree(d_H));
    checkCuda(cudaFree(d_W2));
    checkCuda(cudaFree(d_b2));
    checkCuda(cudaFree(d_Z));

    return (torchFailures == 0 && matchingPredictions == M)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
