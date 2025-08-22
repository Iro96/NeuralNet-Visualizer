import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# 1. Model Definition
# -------------------------------
class TinyWideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def build_model():
    return TinyWideNet()

def build_example_input(model):
    return torch.randn(1, 6)

# -------------------------------
# 2. Dummy Dataset
# -------------------------------
# 6 input features, 3 classes
X = torch.randn(500, 6)
y = torch.randint(0, 3, (500,))  # labels: 0, 1, 2

# -------------------------------
# 3. Training Setup
# -------------------------------
model = build_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# 4. Training Loop
# -------------------------------
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -------------------------------
# 5. Save Model
# -------------------------------
torch.save(model.state_dict(), "./models/tinywidenet.pth")
print("✅ Model saved as tinywidenet.pth")
