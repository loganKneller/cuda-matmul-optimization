#pragma once

#include <cuda_runtime.h>

// All matrices are row-major: C (M x N) = A (M x K) * B (K x N).

// Kernel 1: one thread per output element.
// Consecutive threadIdx.x values map to consecutive columns, so loads of B and
// stores to C are coalesced, while each warp reads the same element of A.
__global__ void naiveGemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 2: shared-memory tiling.
// Each block stages a TILE x TILE tile of A and of B in shared memory, so every
// value loaded from global memory is reused TILE times instead of once.
// With ADD_BIAS, computes C = A * B + bias, adding the length-N bias to each
// output before the single write to C.
constexpr int TILE = 16;

template <bool ADD_BIAS = false>
__global__ void tiledGemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    const float* bias = nullptr
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

        // Out-of-bounds positions are zero-filled so partial edge tiles add nothing.
        tileA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (ADD_BIAS) {
            sum += bias[col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 3: 1D register blocking.
// Each block computes a BM x BN tile of C, and each thread computes TM outputs
// stacked in one column. In the inner loop, a thread loads one value of B into a
// register and reuses it for all TM outputs, so it does TM multiply-adds per B load
// instead of one.
// Launch with BM * BN / TM threads per block (a 1D block).
template <int BM, int BN, int BK, int TM>
__global__ void blocktiling1dGemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    // Top-left corner of this block's tile of C.
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Which outputs this thread computes: TM rows starting at threadRow * TM, in column threadCol.
    int threadCol = threadIdx.x % BN;
    int threadRow = threadIdx.x / BN;

    // Which element of each tile this thread copies from global memory.
    // There are BM * BK elements in tileA and BK * BN in tileB, one per thread.
    int loadRowA = threadIdx.x / BK;
    int loadColA = threadIdx.x % BK;
    int loadRowB = threadIdx.x / BN;
    int loadColB = threadIdx.x % BN;

    float results[TM] = {0.0f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        int aRow = blockRow + loadRowA;
        int aCol = k0 + loadColA;
        int bRow = k0 + loadRowB;
        int bCol = blockCol + loadColB;

        // Out-of-bounds positions are zero-filled so partial edge tiles add nothing.
        tileA[loadRowA][loadColA] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        tileB[loadRowB][loadColB] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float b = tileB[k][threadCol];  // loaded once, reused TM times
            for (int r = 0; r < TM; r++) {
                results[r] += tileA[threadRow * TM + r][k] * b;
            }
        }

        __syncthreads();
    }

    int col = blockCol + threadCol;
    for (int r = 0; r < TM; r++) {
        int row = blockRow + threadRow * TM + r;
        if (row < M && col < N) {
            C[row * N + col] = results[r];
        }
    }
}


/*
 * Kernel 4: 2D register-blocked GEMM
 *
 * Each block computes a BM x BN tile of C.
 * Each thread computes a TM x TN sub-tile.
 *
 * A and B are loaded cooperatively into shared memory.
 * Each thread reuses values from both matrices to update
 * multiple output accumulators held in registers.
 *
 * With BIAS_RELU, computes C = ReLU(A * B + bias) instead, applying the
 * length-N bias and ReLU in registers before the single write to C.
 */
template <int BM, int BN, int BK, int TM, int TN, bool BIAS_RELU = false>
__global__ void blocktiling2dGemm(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N,
    const float* bias = nullptr
) {
    static_assert(BM % TM == 0 && BN % TN == 0,
                  "Tile dimensions must be divisible.");

    constexpr int THREADS_X = BN / TN;
    constexpr int THREADS_Y = BM / TM;
    constexpr int NUM_THREADS = THREADS_X * THREADS_Y;

    // Shared-memory tiles.
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Flatten thread coordinates for cooperative loading.
    int tid = ty * THREADS_X + tx;

    // Top-left corner of the block's output tile.
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Top-left corner of this thread's output tile.
    int threadRow = ty * TM;
    int threadCol = tx * TN;

    // Each thread accumulates TM * TN values.
    float results[TM][TN] = {};

    // Process K in chunks of BK.
    for (int k0 = 0; k0 < K; k0 += BK) {

        // Cooperatively load A into shared memory.
        for (int idx = tid; idx < BM * BK;
             idx += NUM_THREADS) {

            int row = idx / BK;
            int col = idx % BK;

            int globalRow = blockRow + row;
            int globalCol = k0 + col;

            tileA[row][col] =
                (globalRow < M && globalCol < K)
                    ? A[globalRow * K + globalCol]
                    : 0.0f;
        }

        // Cooperatively load B into shared memory.
        for (int idx = tid; idx < BK * BN;
             idx += NUM_THREADS) {

            int row = idx / BN;
            int col = idx % BN;

            int globalRow = k0 + row;
            int globalCol = blockCol + col;

            tileB[row][col] =
                (globalRow < K && globalCol < N)
                    ? B[globalRow * N + globalCol]
                    : 0.0f;
        }

        // All shared-memory loads must finish.
        __syncthreads();

        // Compute partial results using register reuse.
        #pragma unroll
        for (int k = 0; k < BK; k++) {

            float regA[TM];
            float regB[TN];

            // Load TM values from A into registers.
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                regA[r] = tileA[threadRow + r][k];
            }

            // Load TN values from B into registers.
            #pragma unroll
            for (int c = 0; c < TN; c++) {
                regB[c] = tileB[k][threadCol + c];
            }

            // Outer product: TM x TN updates.
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                #pragma unroll
                for (int c = 0; c < TN; c++) {
                    results[r][c] += regA[r] * regB[c];
                }
            }
        }

        // Prevent overwriting tiles before all reads finish.
        __syncthreads();
    }

    // Load the bias for this thread's TN columns once, reused for all TM rows.
    float regBias[TN] = {};
    if (BIAS_RELU) {
        #pragma unroll
        for (int c = 0; c < TN; c++) {
            int col = blockCol + threadCol + c;
            regBias[c] = (col < N) ? bias[col] : 0.0f;
        }
    }

    // Write the final output tile to global memory.
    #pragma unroll
    for (int r = 0; r < TM; r++) {
        int row = blockRow + threadRow + r;

        #pragma unroll
        for (int c = 0; c < TN; c++) {
            int col = blockCol + threadCol + c;

            if (row < M && col < N) {
                float value = results[r][c];
                if (BIAS_RELU) {
                    value = fmaxf(0.0f, value + regBias[c]);
                }
                C[row * N + col] = value;
            }
        }
    }
}
