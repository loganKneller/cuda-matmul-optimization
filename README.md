# cuda-matmul-optimization

Hand-written CUDA matrix multiplication kernels, benchmarked against cuBLAS and
used to run a trained MNIST classifier on the GPU.

The project has two parts:

- **SGEMM kernels** (`gemm/`): a progression of single-precision matrix
  multiplication kernels, from one thread per output to 2D register blocking,
  each checked for correctness and timed against cuBLAS.
- **MLP inference** (`mlp/`): a 784 → 128 → 10 multilayer perceptron trained in
  PyTorch, run with the same kernels, with bias and ReLU fused into the matrix
  multiplications. Every run is validated against PyTorch's output logits.

## Repository layout

```
gemm/
  kernels.cuh          SGEMM kernels shared by both benchmarks
  gemm_bench.cu        times each kernel against cuBLAS on several shapes
mlp/
  mlp_benchmark.cu     MLP inference benchmark (modes, batch sizes, validation)
python/
  train_linear.py      trains a single-layer linear classifier
  train_mlp.py         trains the 784 -> 128 -> 10 MLP
  export_linear.py     exports linear weights and test data for CUDA
  export_mlp.py        exports MLP weights, test data, and reference logits
weights/
  linear/              linear model: PyTorch checkpoint and CUDA binaries
  mlp/                 MLP: PyTorch checkpoint, CUDA binaries, and the full
                       10,000-image test set (full_test/)
archive/               earlier standalone programs, in the order they were built
notebooks/             Colab notebook used to run on a T4 GPU
```

## SGEMM kernels

All matrices are row-major, computing C (M × N) = A (M × K) · B (K × N).

| Kernel | Idea |
|---|---|
| `naiveGemm` | One thread per output element. |
| `tiledGemm` | Each block stages 16 × 16 tiles of A and B in shared memory, reusing each loaded value 16 times. |
| `blocktiling1dGemm` | Each thread computes 8 outputs in a column, reusing one value of B from a register for all 8. |
| `blocktiling2dGemm` | Each thread computes a 4 × 4 sub-tile as an outer product of register-cached values of A and B. |

`tiledGemm` and `blocktiling2dGemm` take an optional compile-time flag that
fuses the bias add (and, for the 2D kernel, ReLU) into the final write, so a
layer's output is written to global memory once.

`gemm_bench` runs every kernel on square matrices from 512 to 4096 and on the
two MLP layer shapes, and reports time, GFLOP/s, percentage of cuBLAS, and
maximum error against cuBLAS. cuBLAS runs in plain FP32 mode (no TF32 tensor
cores) for a like-for-like comparison.

## MLP inference benchmark

The model is `Linear(784, 128) → ReLU → Linear(128, 10)`, trained for 5 epochs.
Its weights are exported as raw float32 files that the CUDA program loads.

`mlp_benchmark` runs the forward pass in one of four configurations:

| Flag | Layer 1 | Layer 2 | Kernel launches |
|---|---|---|---|
| `--naive` | naive GEMM, then bias, then ReLU | naive GEMM, then bias | 5 |
| `--unfused` | 2D register blocking, then bias, then ReLU | shared-memory tiling, then bias | 5 |
| `--fuse-layer1` | 2D register blocking with fused bias + ReLU | shared-memory tiling, then bias | 3 |
| `--fused` (default) | 2D register blocking with fused bias + ReLU | shared-memory tiling with fused bias | 2 |

Layer 2 uses simple tiling rather than 2D blocking because its output is only
10 columns wide, so a 64-column tile would leave most threads idle.

`--batch M` runs on the first M test images (1 to 10,000; default 10,000). The
benchmark reports GPU execution time, host-to-device and device-to-host
transfer times, and end-to-end time, each as batch latency and images per
second, and prints a `CSV,` line for collecting results across runs. Small
batches repeat the forward pass back to back within each measurement so the
timing is above CUDA event resolution, which measures sustained time per batch
rather than the latency of a single isolated request.

Each run checks every output logit against PyTorch (absolute tolerance 1e-3,
relative 1e-4) and exits with an error if any differ or any prediction
disagrees.

## Results

All results are on an NVIDIA Tesla T4 (Google Colab).

- **Accuracy:** 97.38% on the 10,000-image MNIST test set, with predictions
  identical to PyTorch and a maximum logit difference of about 1e-5.
- **Naive baseline:** 6.16 ms of GPU time for all 10,000 images.

<!-- TODO: add the batch-size sweep (throughput vs. batch size for each mode)
     and the gemm_bench comparison against cuBLAS. -->

## Building and running

Requires the CUDA toolkit and an NVIDIA GPU. Build from the repo root:

```bash
make                  # builds ./mlp_benchmark and gemm/gemm_bench
make ARCH=sm_75       # if nvcc does not support -arch=native (e.g. on Colab's T4)
```

Run from the repo root, since `mlp_benchmark` loads `weights/` relative to the
working directory:

```bash
./mlp_benchmark                      # fully fused, all 10,000 images
./mlp_benchmark --naive --batch 64   # naive baseline on the first 64 images
./gemm/gemm_bench                    # all kernels vs. cuBLAS
./gemm/gemm_bench 4096 4096 4096     # a single custom shape
```

To sweep batch sizes for every mode:

```bash
for mode in --naive --unfused --fuse-layer1 --fused; do
  for b in 1 64 256 1024 10000; do
    ./mlp_benchmark $mode --batch $b | grep '^CSV,'
  done
done > batch_sweep.csv
```

## Retraining the model

The exported weights are committed, so this is only needed to retrain. With
PyTorch and torchvision installed:

```bash
python python/train_mlp.py     # downloads MNIST to data/, saves weights/mlp/mlp_mnist.pt
python python/export_mlp.py    # writes the CUDA binaries to weights/mlp/
```

`train_linear.py` and `export_linear.py` do the same for the single-layer
linear model in `weights/linear/`.

## Project history

`archive/` keeps the earlier standalone programs from building up to this
point, numbered in order: a vector add, a CPU matrix multiply, a first CUDA
matrix multiply, a reader for the exported test data, linear and MLP
inference, and the first tiled MLP inference.
Their kernels have since moved into `gemm/kernels.cuh` and their benchmarking
into `mlp/mlp_benchmark.cu`. Like `mlp_benchmark`, they expect to be run from
the repo root.
