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

// Benchmark configurations, from the original baseline to fully fused.
enum class Mode {
    Naive,       // naive GEMMs, separate bias/ReLU kernels (5 launches)
    Unfused,     // shape-optimized GEMMs, separate bias/ReLU kernels (5 launches)
    FuseLayer1,  // layer 1 bias + ReLU fused into its GEMM (3 launches)
    Fused,       // both layers' epilogues fused into their GEMMs (2 launches)
};

int main(int argc, char** argv) {
    const int MAX_IMAGES = 10000;  // Images in the exported MNIST test set

    // By default, run the fully fused, shape-optimized forward pass on all
    // test images. The other modes reproduce earlier configurations, and
    // --batch M runs on only the first M images.
    Mode mode = Mode::Fused;
    int M = MAX_IMAGES;  // Batch size
    for (int i = 1; i < argc; i++) {
        const string arg = argv[i];
        if (arg == "--fused") {
            mode = Mode::Fused;
        } else if (arg == "--fuse-layer1") {
            mode = Mode::FuseLayer1;
        } else if (arg == "--unfused") {
            mode = Mode::Unfused;
        } else if (arg == "--naive") {
            mode = Mode::Naive;
        } else if (arg == "--batch" && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else {
            M = 0;  // Unrecognized argument: fall through to the usage message.
            break;
        }
    }
    if (M < 1 || M > MAX_IMAGES) {
        cerr << "Usage: " << argv[0]
             << " [--fused | --fuse-layer1 | --unfused | --naive] [--batch M]" << endl
             << "  M is the number of images, from 1 to " << MAX_IMAGES
             << " (default " << MAX_IMAGES << ")" << endl;
        return EXIT_FAILURE;
    }
    const bool fuseLayer1 = (mode == Mode::FuseLayer1 || mode == Mode::Fused);
    const bool fuseLayer2 = (mode == Mode::Fused);

    const int INPUT = 784;    // Pixels per image
    const int HIDDEN = 128;   // Neurons in hidden layer
    const int OUTPUT = 10;    // Digit scores per image

    // Host-side model inputs, parameters, and outputs. Per-image arrays are
    // sized for M images, so loadBinary reads the first M images of each file.
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
    // Layer 1 (Mx784 times 784x128): 2D register blocking, BK=16.
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    const dim3 block1(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    const dim3 grid1((HIDDEN + BN - 1) / BN, (M + BM - 1) / BM);

    // Layer 2 (Mx128 times 128x10): simple shared-memory tiling.
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

    const int launches = 2 + (fuseLayer1 ? 0 : 2) + (fuseLayer2 ? 0 : 1);
    string modeName;
    cout << "Inference mode: ";
    switch (mode) {
        case Mode::Naive:      modeName = "naive";       cout << "naive GEMM baseline"; break;
        case Mode::Unfused:    modeName = "unfused";     cout << "shape-optimized GEMM, no fusion"; break;
        case Mode::FuseLayer1: modeName = "fuse-layer1"; cout << "shape-optimized GEMM, layer 1 fused"; break;
        case Mode::Fused:      modeName = "fused";       cout << "shape-optimized GEMM, both layers fused"; break;
    }
    cout << " (" << launches << " kernel launches)" << endl;
    cout << "Batch size: " << M << " images" << endl;
    if (mode != Mode::Naive) {
        cout << "  Layer 1: 2D register blocking (BM=64, BN=64, BK=16, TM=4, TN=4)"
             << (fuseLayer1 ? " + fused bias/ReLU" : "") << endl;
        cout << "  Layer 2: shared-memory tiling (TILE=" << TILE << ")"
             << (fuseLayer2 ? " + fused bias" : "") << endl;
    }

    // Fused GEMMs apply bias (and ReLU) in registers before their single write,
    // replacing the separate addBias/relu kernels and their extra memory passes.
    // All kernels execute in the default CUDA stream, in launch order.
    auto runInference = [&]() {
        // Hidden layer: H = ReLU(X * W1 + b1)
        if (mode == Mode::Naive) {
            naiveGemm<<<naiveGrid1, naiveBlock>>>(
                d_X, d_W1, d_H, M, INPUT, HIDDEN
            );
        } else if (fuseLayer1) {
            blocktiling2dGemm<BM, BN, BK, TM, TN, true><<<grid1, block1>>>(
                d_X, d_W1, d_H, M, INPUT, HIDDEN, d_b1
            );
        } else {
            blocktiling2dGemm<BM, BN, BK, TM, TN><<<grid1, block1>>>(
                d_X, d_W1, d_H, M, INPUT, HIDDEN
            );
        }
        checkCuda(cudaGetLastError());

        if (!fuseLayer1) {
            addBias<<<hiddenBlocks, threads>>>(d_H, d_b1, M, HIDDEN);
            checkCuda(cudaGetLastError());

            relu<<<hiddenBlocks, threads>>>(d_H, hiddenSize);
            checkCuda(cudaGetLastError());
        }

        // Output layer: Z = H * W2 + b2
        if (mode == Mode::Naive) {
            naiveGemm<<<naiveGrid2, naiveBlock>>>(
                d_H, d_W2, d_Z, M, HIDDEN, OUTPUT
            );
        } else if (fuseLayer2) {
            tiledGemm<true><<<grid2, block2>>>(
                d_H, d_W2, d_Z, M, HIDDEN, OUTPUT, d_b2
            );
        } else {
            tiledGemm<false><<<grid2, block2>>>(
                d_H, d_W2, d_Z, M, HIDDEN, OUTPUT
            );
        }
        checkCuda(cudaGetLastError());

        if (!fuseLayer2) {
            addBias<<<outputBlocks, threads>>>(d_Z, d_b2, M, OUTPUT);
            checkCuda(cudaGetLastError());
        }
    };

    const int warmupRuns = 5;
    const int measuredRuns = 30;

    // Small batches finish in microseconds, near the resolution of CUDA event
    // timing. Each measurement therefore repeats the operation back to back,
    // covering roughly MAX_IMAGES images, and divides by the repetition count.
    // Back-to-back launches overlap CPU launch cost with GPU work, so this is
    // sustained time per batch, not the latency of one isolated request.
    const int reps = max(1, MAX_IMAGES / M);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // Median time in ms of one call to op, after warmupRuns untimed calls.
    auto medianMs = [&](auto op) {
        for (int i = 0; i < warmupRuns; i++) {
            op();
        }
        checkCuda(cudaDeviceSynchronize());

        vector<float> times;
        times.reserve(measuredRuns);

        for (int i = 0; i < measuredRuns; i++) {
            checkCuda(cudaEventRecord(start));
            for (int r = 0; r < reps; r++) {
                op();
            }
            checkCuda(cudaEventRecord(stop));
            checkCuda(cudaEventSynchronize(stop));

            float elapsedMs = 0.0f;
            checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop));
            times.push_back(elapsedMs / reps);
        }

        sort(times.begin(), times.end());
        return (times[measuredRuns / 2 - 1] + times[measuredRuns / 2]) / 2.0f;
    };

    // GPU execution: the kernels of one forward pass, with inputs already on
    // the GPU. Excludes allocations, data loading, and correctness checks.
    const float computeMs = medianMs(runInference);

    // Transfers of one batch's per-image data, timed separately from GPU
    // execution. Host arrays are pageable (std::vector), not pinned. Weights
    // are copied once at startup and excluded, as in a deployed model.
    const float h2dMs = medianMs([&]() {
        checkCuda(cudaMemcpy(
            d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice
        ));
    });
    const float d2hMs = medianMs([&]() {
        checkCuda(cudaMemcpy(
            h_Z.data(), d_Z, h_Z.size() * sizeof(float), cudaMemcpyDeviceToHost
        ));
    });

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    const float endToEndMs = h2dMs + computeMs + d2hMs;
    const double computeImagesPerSec = M / (computeMs / 1000.0);
    const double endToEndImagesPerSec = M / (endToEndMs / 1000.0);

    cout << fixed << setprecision(4);
    cout << "Timing: median of " << measuredRuns << " measurements of "
         << reps << " repetitions each, after " << warmupRuns << " warm-up runs" << endl;
    cout << "GPU execution (kernels only):" << endl;
    cout << "  Batch latency: " << computeMs << " ms" << endl;
    cout << "  Throughput: " << setprecision(0) << computeImagesPerSec
         << " images/s" << setprecision(4) << endl;
    cout << "Host/device transfers (pageable memory):" << endl;
    cout << "  Input H2D (" << h_X.size() * sizeof(float) << " bytes): "
         << h2dMs << " ms" << endl;
    cout << "  Output D2H (" << h_Z.size() * sizeof(float) << " bytes): "
         << d2hMs << " ms" << endl;
    cout << "End-to-end (H2D + GPU execution + D2H):" << endl;
    cout << "  Batch latency: " << endToEndMs << " ms" << endl;
    cout << "  Throughput: " << setprecision(0) << endToEndImagesPerSec
         << " images/s" << setprecision(4) << endl;

    // One machine-readable line per run, for plotting throughput vs. batch size.
    cout << "CSV columns: mode,batch,reps,gpu_ms,h2d_ms,d2h_ms,gpu_images_per_s,e2e_images_per_s" << endl;
    cout << "CSV," << modeName << "," << M << "," << reps << ","
         << setprecision(6) << computeMs << "," << h2dMs << "," << d2hMs << ","
         << setprecision(0) << computeImagesPerSec << "," << endToEndImagesPerSec
         << setprecision(4) << endl;

    // Transfer the final output for validation (excluded from timing).
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
