#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
    const int M = 64;
    const int K = 784;

    vector<float> inputs(M * K);

    ifstream file("weights/linear/inputs_cuda.bin", ios::binary);
    if (!file.is_open()) {
        cerr << "Could not open inputs_cuda.bin" << endl;
        return 1;
    }

    file.read(
        reinterpret_cast<char*>(inputs.data()),
        inputs.size() * sizeof(float)
    );

    if (!file) {
        cerr << "Failed to read all input values" << endl;
        return 1;
    }

    cout << "Successfully loaded "
         << inputs.size()
         << " pixel values" << endl;


const int N = 10;

// Allocate memory for weights and biases.
vector<float> weights(K * N);
vector<float> biases(N);

// Open the exported binary files.
ifstream weightFile("weights/linear/weights_cuda.bin", ios::binary);
ifstream biasFile("weights/linear/biases_cuda.bin", ios::binary);

if (!weightFile || !biasFile) {
    cerr << "Could not open weights or biases!" << endl;
    return 1;
}

    // Load the values into our vectors.
    weightFile.read(
        reinterpret_cast<char*>(weights.data()),
        weights.size() * sizeof(float)
    );

    biasFile.read(
        reinterpret_cast<char*>(biases.data()),
        biases.size() * sizeof(float)
    );

    if (!weightFile || !biasFile) {
        cerr << "Failed to read weights or biases :(" << endl;
        return 1;
    }

    cout << "Loaded " << weights.size() << " weights." << endl;
    cout << "Loaded " << biases.size() << " biases." << endl;

    cout << "First weight: " << weights[0] << endl;
    cout << "First bias: " << biases[0] << endl;

    return 0;
}
