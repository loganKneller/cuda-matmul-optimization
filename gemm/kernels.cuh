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
constexpr int TILE = 16;

__global__ void tiledGemm(
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
        C[row * N + col] = sum;
    }
}
