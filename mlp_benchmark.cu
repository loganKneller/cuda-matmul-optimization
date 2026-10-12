#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>

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


__global__ void relu(float* values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}


bool loadBinary(
    const string& path,
    vector<float>& destination
) {
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
        cerr << "Could not read all values from "
             << path << endl;
        return false;
    }

    return true;
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
    const int M = 10000; //number of images
    const int INPUT = 784; //pixels per image
    const int HIDDEN = 128; //number of hidden neurons
    const int OUTPUT = 10; //number of outputs (digits 0-9)

        
    
    vector<float> h_X(M * INPUT);          // 10000 x 784
    vector<float> h_W1(INPUT * HIDDEN);   // 784 x 128
    vector<float> h_b1(HIDDEN);           // 128 biases

    vector<float> h_W2(HIDDEN * OUTPUT);  // 128 x 10
    vector<float> h_b2(OUTPUT);          // 10 biases

    vector<float> h_Z(M * OUTPUT);       // 10000 x 10


    // reading exported MNIST files
    
    if (
        !loadBinary("weights/mlp/full_test/inputs_cuda.bin", h_X) ||
        !loadBinary("weights/mlp/weights1_cuda.bin", h_W1) ||
        !loadBinary("weights/mlp/biases1_cuda.bin", h_b1) ||
        !loadBinary("weights/mlp/weights2_cuda.bin", h_W2) ||
        !loadBinary("weights/mlp/biases2_cuda.bin", h_b2)
    ) {
        return 1;
    }

    cout << "Successfully loaded all MLP data" << endl;


    // now, load pytorch reference output
    
    vector<float> reference(M * OUTPUT);

    if (!loadBinary(
        "weights/mlp/full_test/reference_logits.bin",
        reference
    )) {
        return 1;
    }
    cout << "Successfully loaded PyTorch reference" << endl;


    // load the correct MNIST files.
    // we can't use loadBinary because it expecs floats, this uses int32_t
    vector<int32_t> labels(M);

    ifstream labelFile(
        "weights/mlp/full_test/labels.bin",
        ios::binary
    );

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

    cout << "Successfully loaded "
         << labels.size()
         << " MNIST labels" << endl;

    // device pointers
    
    float* d_X = nullptr;
    float* d_W1 = nullptr;
    float* d_b1 = nullptr;
    float* d_H = nullptr;

    float* d_W2 = nullptr;
    float* d_b2 = nullptr;
    float* d_Z = nullptr;

    
    // allocate memory for the input and first layes
    checkCuda(cudaMalloc((void**)&d_X, h_X.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_W1, h_W1.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b1, h_b1.size() * sizeof(float)));

    // allocate memory for hidden-layer output
    checkCuda(cudaMalloc((void**)&d_H, M * HIDDEN * sizeof(float)));

    // allocate memory for second layer and final output
    checkCuda(cudaMalloc((void**)&d_W2, h_W2.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b2, h_b2.size() * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_Z, h_Z.size() * sizeof(float)));


   
    // copy input images to GPU
    checkCuda(cudaMemcpy(
        d_X, h_X.data(),
        h_X.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // copy first layer parameters
    checkCuda(cudaMemcpy(
        d_W1, h_W1.data(),
        h_W1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    checkCuda(cudaMemcpy(
        d_b1, h_b1.data(),
        h_b1.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // second layer parameters
    checkCuda(cudaMemcpy(
        d_W2, h_W2.data(),
        h_W2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    checkCuda(cudaMemcpy(
        d_b2, h_b2.data(),
        h_b2.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    cout << "Successfully copied MLP data to GPU!" << endl;

    // initializing block of threads
    // most efficient size depends on the size of input matrix
    dim3 block(16, 16);

    dim3 grid1(
        (HIDDEN + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    cudaEvent_t start, stop;

    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // start measuring GPU execution time
    checkCuda(cudaEventRecord(start));

    // d_H = X * W1
    matmul<<<grid1, block>>>(
        d_X, d_W1, d_H,
        M, INPUT, HIDDEN
    );

    checkCuda(cudaGetLastError());

    // add the learned biases
    
    // number of elements in the hidden matrix
    int hiddenSize = M * HIDDEN;

    // configure a 1D kernel launch
    int threads = 256;
    int blocks = (hiddenSize + threads - 1) / threads;

    // add the first layer's biases
    addBias<<<blocks, threads>>>(
        d_H, d_b1, M, HIDDEN
    );
    checkCuda(cudaGetLastError());

    // apply ReLU to the hidden-layer values
    relu<<<blocks, threads>>>(d_H, hiddenSize);
    checkCuda(cudaGetLastError());

    checkCuda(cudaDeviceSynchronize());

    cout << "Hidden layer computed successfully" << endl;

    // configure the second matrix multiplication
    dim3 grid2(
        (OUTPUT + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // hidden activations * second weight matrix
    matmul<<<grid2, block>>>(
        d_H, d_W2, d_Z,
        M, HIDDEN, OUTPUT
    );

    checkCuda(cudaGetLastError());


    // total number of values in the final output matrix
    int outputSize = M * OUTPUT;

    // 1D launch for bias addition
    int outputBlocks = (outputSize + threads - 1) / threads;
    addBias<<<outputBlocks, threads>>>(
        d_Z, d_b2, M, OUTPUT
    );

    checkCuda(cudaGetLastError());


    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;

    checkCuda(cudaEventElapsedTime(
        &elapsedMs, start, stop
    ));

    cout << "CUDA inference time for "
        << M << " images: "
        << elapsedMs << " ms" << endl;

    cout << "Average GPU time per image: "
        << elapsedMs / M << " ms" << endl;

    // Release timing resources.
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    checkCuda(cudaDeviceSynchronize());

    //At this point, our forward pass is done!!!
    cout << "MLP forward pass completed on GPU!" << endl;


    // copy result from GPU back to CPU
    checkCuda(cudaMemcpy(
        h_Z.data(),
        d_Z,
        h_Z.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cout << "Successfully copied MLP output to CPU" << endl;

    // compare CUDA output against PyTorch
    const float atol = 1e-3f;
    const float rtol = 1e-4f;
    int torchFailures = 0;
    int matchingPredictions = 0;
    float maxTorchError = 0.0f;
    int correctPredictions = 0;

    for (int i = 0; i < M; i++) {
        int gpuPrediction = 0;
        int torchPrediction = 0;

        for (int j = 0; j < OUTPUT; j++) {
            int idx = i * OUTPUT + j;

            float gpuValue = h_Z[idx];
            float torchValue = reference[idx];

            // compare the numerical outputs
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

            // find the highest-scoring digit
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

    double accuracy = 100.0 * correctPredictions / M;

    cout << "Correct predictions: "
        << correctPredictions << "/" << M << endl;

    cout << "CUDA MNIST accuracy: "
        << accuracy << "%" << endl;


    checkCuda(cudaFree(d_X));
    checkCuda(cudaFree(d_W1));
    checkCuda(cudaFree(d_b1));
    checkCuda(cudaFree(d_H));
    checkCuda(cudaFree(d_W2));
    checkCuda(cudaFree(d_b2));
    checkCuda(cudaFree(d_Z));

    return torchFailures == 0 ? 0 : 1;
}
