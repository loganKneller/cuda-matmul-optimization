import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# data/ and weights/ live at the repo root, one level above this script
BASE_DIR = Path(__file__).resolve().parent.parent

torch.manual_seed(100)

# step 1: load the datasets
train_data = datasets.MNIST(
    root=BASE_DIR / "data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root=BASE_DIR / "data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)

# define the model
model = nn.Linear(784, 10)

# define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train for 5 epochs
for epoch in range(5):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        # flatten 28x28 images into 784-element vectors
        X = images.reshape(-1, 784)

        # forward pass
        scores = model(X)
        loss = loss_fn(scores, labels)

        # backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    avg_loss = total_loss / len(train_data)
    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

# evaluate on unseen test images
model.eval()
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        X = images.reshape(-1, 784)
        scores = model(X)

        predictions = scores.argmax(dim=1)
        correct += (predictions == labels).sum().item()

accuracy = correct / len(test_data)
print(f"Test accuracy: {accuracy:.2%}")

# save the trained parameters
weights_dir = BASE_DIR / "weights" / "linear"
weights_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    weights_dir / "linear_mnist.pt"
)

print("Model saved!")
print("Weight shape:", model.weight.shape)
print("Bias shape:", model.bias.shape)
