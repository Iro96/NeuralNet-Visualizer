import torch
import torch.nn as nn
import torch.optim as optim

class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 20)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = TinyAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Correct input dimension
data = torch.randn(10000, 20)

num_epochs = 500
batch_size = 32
best_loss = float('inf')

for epoch in range(num_epochs):
    permutation = torch.randperm(data.size(0))
    epoch_loss = 0.0

    for i in range(0, data.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = data[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (data.size(0) // batch_size)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "./models/tiny_autoencoder.pth")

# Test reconstruction
test_input = torch.randn(1, 20)
reconstructed = model(test_input)
print("Original:", test_input)
print("Reconstructed:", reconstructed)
