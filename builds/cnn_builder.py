# train_tinycnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Model ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 28 * 28, 10)
        )
    def forward(self, x):
        return self.net(x)

def build_model():
    return TinyCNN()

def build_example_input(model):
    return torch.randn(1, 1, 28, 28)

# --- Training ---
def train_and_save(epochs=20, batch_size=64, lr=0.01, save_path="./models/tinycnn.pth"):
    # MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")

if __name__ == "__main__":
    train_and_save()
