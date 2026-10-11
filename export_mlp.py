from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# load parameters
parameters = torch.load(
    BASE_DIR / "weights" / "mlp_mnist.pt",
    map_location="cpu",
    weights_only=True
)

# extract layer 1 weight matrix
W1 = parameters["0.weight"]

# transpose into a form readable by the .cu program
W1_cuda = W1.T.contiguous()

# export weights
output_dir = BASE_DIR / "weights" / "mlp"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "weights1_cuda.bin"
W1_cuda.numpy().tofile(output_path)

# extract/export layer 1 biases
b1 = parameters["0.bias"]
bias_path = output_dir / "biases1_cuda.bin"
b1.numpy().tofile(bias_path)

# extract layer 2 weights
W2 = parameters["2.weight"]

W2_cuda = W2.T.contiguous()

# export weights
output_path = output_dir / "weights2_cuda.bin"
W2_cuda.numpy().tofile(output_path)

# extract/export layer 2 biases
b2 = parameters["2.bias"]
bias_path = output_dir / "biases2_cuda.bin"
b2.numpy().tofile(bias_path)

# load MNIST test data set, and select the first 64 images
test_data = datasets.MNIST(
    root=BASE_DIR / "data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(test_loader))

# flatten the images, then export
X = images.reshape(64, 784)

input_path = output_dir / "inputs_cuda.bin"
X.numpy().tofile(input_path)


# now, calculate MLP's output with Pytorch for comparison
with torch.no_grad():
    H = torch.relu(
        torch.nn.functional.linear(X, W1, b1)
    )

    reference_logits = torch.nn.functional.linear(H, W2, b2)

# export reference outputs
reference_path = output_dir / "reference_logits.bin"
reference_logits.numpy().tofile(reference_path)

predictions = reference_logits.argmax(dim=1)

print("Reference shape:", reference_logits.shape)
print("Reference file size:", reference_path.stat().st_size, "bytes")
print("First 10 predictions:", predictions[:10].tolist())
print("First 10 true labels:", labels[:10].tolist())

# Calculate accuracy on these 64 images.
accuracy = (predictions == labels).float().mean().item()
print(f"Batch accuracy: {accuracy:.2%}")
