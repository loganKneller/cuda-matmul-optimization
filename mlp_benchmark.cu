#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <iomanip>

using namespace std;

// Exit if a CUDA runtime call fails.
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// C = A * B, with row-major matrices.
__global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

constexpr int TILE = 16;

__global__ void tiledMatmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {

    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        tileA[ty][tx] =
            (row < M && aCol < K)
                ? A[row * K + aCol]
                : 0.0f;

        tileB[ty][tx] =
            (bRow < K && col < N)
                ? B[bRow * N + col]
                : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

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

int main() {
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

    // Compute all kernel launch configurations once.
    const dim3 block(16, 16);
    const dim3 grid1(
        (HIDDEN + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );
    const dim3 grid2(
        (OUTPUT + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    const int threads = 256;
    const int hiddenSize = M * HIDDEN;
    const int hiddenBlocks = (hiddenSize + threads - 1) / threads;
    const int outputSize = M * OUTPUT;
    const int outputBlocks = (outputSize + threads - 1) / threads;

    // Reusable GPU-only forward pass. All launches use the default stream.
    // [&] lets this lambda access the device pointers and launch configurations above.
    auto runInference = [&]() {
        // Hidden layer: H = ReLU(X * W1 + b1)
        matmul<<<grid1, block>>>(d_X, d_W1, d_H, M, INPUT, HIDDEN);
        checkCuda(cudaGetLastError());

        addBias<<<hiddenBlocks, threads>>>(d_H, d_b1, M, HIDDEN);
        checkCuda(cudaGetLastError());

        relu<<<hiddenBlocks, threads>>>(d_H, hiddenSize);
        checkCuda(cudaGetLastError());

        // Output layer: Z = H * W2 + b2
        matmul<<<grid2, block>>>(d_H, d_W2, d_Z, M, HIDDEN, OUTPUT);
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

    // Now measure the entire five-kernel forward pass repeatedly.
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
    cout << "Amortized median time per image: " << medianMs / M << " ms" << endl;

    // One output transfer after the final measured run, excluded from timing.
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
