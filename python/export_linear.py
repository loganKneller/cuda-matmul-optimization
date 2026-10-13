import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent

# loads the saved parameters from training
parameters = torch.load(
    BASE_DIR / "weights" / "linear" / "linear_mnist.pt",
    map_location="cpu",
    weights_only=True
)

# extracts w's and b's
W = parameters["weight"]
b = parameters["bias"]

# transpose the weights to correspond to CUDA matmult
W_cuda = W.T.contiguous()

# save the transposed weights (output: 32 bit floats)
output_path = BASE_DIR / "weights" / "linear" / "weights_cuda.bin"
W_cuda.numpy().tofile(output_path)

# save biases
bias_path = BASE_DIR / "weights" / "linear" / "biases_cuda.bin"
b.numpy().tofile(bias_path)


# Load MNIST test images.
test_data = datasets.MNIST(
    root=BASE_DIR / "data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# take the first 64 images to use with CUDA
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(test_loader))

# flatten each image into a vector 
# this is necessary because this is what our kernel expects
X = images.reshape(64, 784)

# export the input matrix
input_path = BASE_DIR / "weights" / "linear" / "inputs_cuda.bin"
X.numpy().tofile(input_path)


# calculate reference outputs with pytorch
with torch.no_grad():
    reference_logits = torch.nn.functional.linear(X, W, b)

reference_path = BASE_DIR / "weights" / "linear" / "reference_logits.bin"
reference_logits.numpy().tofile(reference_path)

# determine predicted digits for all of our test images
predictions = reference_logits.argmax(dim=1)

print("Reference shape:", reference_logits.shape)
print("Reference file size:", reference_path.stat().st_size)
print("First 10 predictions:", predictions[:10].tolist())
print("First 10 true labels:", labels[:10].tolist())
