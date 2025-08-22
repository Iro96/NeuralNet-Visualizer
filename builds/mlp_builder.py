import torch
import torch.nn as nn
import torch.optim as optim

# Your model definition
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def forward(self, x):
        return self.net(x)

def build_model():
    return TinyMLP()

def build_example_input(model):
    return torch.randn(1, 10)

# --------------------------
# Training setup
# --------------------------
# Hyperparameters
learning_rate = 0.001
num_epochs = 2000
batch_size = 32

# Create model
model = build_model()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create synthetic dataset
# Inputs: 1000 samples, 10 features
# Targets: 1000 samples, 5 outputs
X = torch.randn(1000, 10)
y = torch.randn(1000, 5)

# Training loop
for epoch in range(num_epochs):
    permutation = torch.randperm(X.size()[0])
    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], y[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "./models/tiny_mlp.pth")
print("Model saved as tiny_mlp.pth")
