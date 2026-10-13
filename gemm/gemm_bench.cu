// SGEMM benchmark: times each custom kernel against cuBLAS on the same inputs.
//
// Usage:
//   ./gemm_bench            run the default shape list
//   ./gemm_bench M K N      run a single custom shape
//
// Prints a table per shape and writes every result to results.csv.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kernels.cuh"

using namespace std;

// Exit if a CUDA runtime call fails.
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// Exit if a cuBLAS call fails.
void checkCublas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS error: " << cublasGetStatusString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// Every kernel is launched through the same signature: C = A * B, row-major.
using LaunchFn = function<void(const float* A, const float* B, float* C, int M, int K, int N)>;

struct Kernel {
    string name;
    LaunchFn launch;
};

struct Shape {
    string label;
    int M, K, N;
};

const int warmupRuns = 3;
const int measuredRuns = 21;  // odd, so the median is a single measured run

// Median time of one launch in milliseconds, after warm-up.
float medianTimeMs(const Kernel& kernel, const float* A, const float* B, float* C, const Shape& s) {
    for (int i = 0; i < warmupRuns; i++) {
        kernel.launch(A, B, C, s.M, s.K, s.N);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    vector<float> times(measuredRuns);
    for (int i = 0; i < measuredRuns; i++) {
        checkCuda(cudaEventRecord(start));
        kernel.launch(A, B, C, s.M, s.K, s.N);
        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaEventSynchronize(stop));
        checkCuda(cudaGetLastError());
        checkCuda(cudaEventElapsedTime(&times[i], start, stop));
    }

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    sort(times.begin(), times.end());
    return times[measuredRuns / 2];
}

// Spot-check the cuBLAS result against double-precision CPU dot products.
// This guards the row-major/column-major conversion in the cuBLAS launcher,
// since cuBLAS is then used as the reference for every other kernel.
double cublasSpotCheckError(
    const vector<float>& A, const vector<float>& B, const vector<float>& C, const Shape& s
) {
    mt19937 rng(123);
    uniform_int_distribution<int> pickRow(0, s.M - 1);
    uniform_int_distribution<int> pickCol(0, s.N - 1);

    // Random positions plus the four corners, where edge-handling bugs show up.
    vector<pair<int, int>> positions = {
        {0, 0}, {0, s.N - 1}, {s.M - 1, 0}, {s.M - 1, s.N - 1}
    };
    for (int i = 0; i < 256; i++) {
        positions.push_back({pickRow(rng), pickCol(rng)});
    }

    double maxErr = 0.0;
    for (auto [row, col] : positions) {
        double expected = 0.0;
        for (int k = 0; k < s.K; k++) {
            expected += (double)A[(size_t)row * s.K + k] * B[(size_t)k * s.N + col];
        }
        double err = fabs(expected - C[(size_t)row * s.N + col]);
        if (!(err <= maxErr)) maxErr = isnan(err) ? INFINITY : err;
    }
    return maxErr;
}

// Inputs are uniform in [-1, 1], so FP32 rounding error grows with K. Wrong
// indexing produces errors many orders of magnitude larger than this bound.
double tolerance(int K) {
    return 1e-5 * K;
}

int main(int argc, char** argv) {
    vector<Shape> shapes;
    if (argc == 4) {
        shapes.push_back({"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3])});
    } else if (argc == 1) {
        shapes = {
            {"square-512", 512, 512, 512},
            {"square-1024", 1024, 1024, 1024},
            {"square-2048", 2048, 2048, 2048},
            {"square-4096", 4096, 4096, 4096},
            // The two layers of the MNIST MLP over the full 10,000-image test set.
            {"mlp-layer1", 10000, 784, 128},
            {"mlp-layer2", 10000, 128, 10},
        };
    } else {
        cerr << "Usage: " << argv[0] << " [M K N]" << endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    cout << "Device: " << prop.name << " (" << prop.multiProcessorCount
         << " SMs, compute " << prop.major << "." << prop.minor << ")" << endl;

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
    // Plain FP32 math, so cuBLAS is compared like-for-like (no TF32 tensor cores).
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // cuBLAS goes first: it is both the performance baseline and the correctness reference.
    vector<Kernel> kernels = {
        {"cuBLAS", [&](const float* A, const float* B, float* C, int M, int K, int N) {
            // cuBLAS is column-major. A row-major M x N matrix has the same memory
            // layout as a column-major N x M matrix, so we compute C^T = B^T * A^T.
            const float alpha = 1.0f;
            const float beta = 0.0f;
            checkCublas(cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, B, N, A, K,
                &beta, C, N
            ));
        }},
        {"naive", [](const float* A, const float* B, float* C, int M, int K, int N) {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            naiveGemm<<<grid, block>>>(A, B, C, M, K, N);
        }},
        {"tiled", [](const float* A, const float* B, float* C, int M, int K, int N) {
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            tiledGemm<<<grid, block>>>(A, B, C, M, K, N);
        }},
        {"1d-block", [](const float* A, const float* B, float* C, int M, int K, int N) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 block(BM * BN / TM);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            blocktiling1dGemm<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, K, N);
        }},
    };

    ofstream csv("results.csv");
    csv << "device,shape,M,K,N,kernel,median_ms,gflops,pct_of_cublas,max_abs_err,passed\n";

    bool allPassed = true;

    for (const Shape& s : shapes) {
        size_t sizeA = (size_t)s.M * s.K;
        size_t sizeB = (size_t)s.K * s.N;
        size_t sizeC = (size_t)s.M * s.N;

        // Same random inputs for every kernel.
        mt19937 rng(42);
        uniform_real_distribution<float> dist(-1.0f, 1.0f);
        vector<float> h_A(sizeA), h_B(sizeB);
        for (float& x : h_A) x = dist(rng);
        for (float& x : h_B) x = dist(rng);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        checkCuda(cudaMalloc((void**)&d_A, sizeA * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_B, sizeB * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_C, sizeC * sizeof(float)));
        checkCuda(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

        cout << "\n" << s.label << "  (M=" << s.M << ", K=" << s.K << ", N=" << s.N << ")" << endl;
        cout << left << setw(12) << "kernel"
             << right << setw(12) << "time (ms)"
             << setw(12) << "GFLOPS"
             << setw(12) << "% cuBLAS"
             << setw(14) << "max abs err"
             << "  check" << endl;

        const double flops = 2.0 * s.M * s.K * s.N;
        vector<float> reference(sizeC), h_C(sizeC);
        float cublasMs = 0.0f;

        for (const Kernel& kernel : kernels) {
            // Poison the output so a kernel that skips elements cannot pass.
            checkCuda(cudaMemset(d_C, 0xFF, sizeC * sizeof(float)));

            float ms = medianTimeMs(kernel, d_A, d_B, d_C, s);
            checkCuda(cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

            double maxErr = 0.0;
            if (kernel.name == "cuBLAS") {
                cublasMs = ms;
                reference = h_C;
                maxErr = cublasSpotCheckError(h_A, h_B, h_C, s);
            } else {
                for (size_t i = 0; i < sizeC; i++) {
                    double err = fabs((double)h_C[i] - reference[i]);
                    // NaN (e.g. from the poison value) never compares greater, so catch it explicitly.
                    if (!(err <= maxErr)) maxErr = isnan(err) ? INFINITY : err;
                }
            }

            bool passed = maxErr <= tolerance(s.K);
            allPassed = allPassed && passed;

            double gflops = flops / (ms * 1e-3) / 1e9;
            double pctOfCublas = 100.0 * cublasMs / ms;

            cout << left << setw(12) << kernel.name << right << fixed
                 << setw(12) << setprecision(4) << ms
                 << setw(12) << setprecision(1) << gflops
                 << setw(11) << setprecision(1) << pctOfCublas << "%"
                 << setw(14) << scientific << setprecision(2) << maxErr
                 << "  " << (passed ? "PASS" : "FAIL") << endl;

            csv << prop.name << "," << s.label << "," << s.M << "," << s.K << "," << s.N << ","
                << kernel.name << "," << ms << "," << gflops << "," << pctOfCublas << ","
                << maxErr << "," << (passed ? 1 : 0) << "\n";
        }

        checkCuda(cudaFree(d_A));
        checkCuda(cudaFree(d_B));
        checkCuda(cudaFree(d_C));
    }

    checkCublas(cublasDestroy(handle));

    cout << "\nResults written to results.csv" << endl;
    if (!allPassed) {
        cerr << "One or more kernels FAILED the correctness check" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
