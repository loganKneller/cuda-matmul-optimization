#include <iostream>
#include <vector>

using namespace std;
// multiplies A and B to yield C
// A is M by K, B is K by N, C is M by N
// A, B, and C are represented as arrays for threading purposes
void matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {

    // implement algorithm

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++){
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

int main() {
    int m = 4;
    int k = 5;
    int n = 3;
    vector<float> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,200};
    vector<float> B = {9,8,7,6,5,4,3,2,1,0,0,1,2,2,3};
    vector<float> C(m * n);

    matmul(A.data(), B.data(), C.data(), m, k, n);

    for (int i = 0; i < 12; i++) {
        std::cout << C[i] << " ";
    }

    std::cout << "\n";
    return 0;
}