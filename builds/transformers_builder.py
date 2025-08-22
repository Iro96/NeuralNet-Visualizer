# transformer_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---- Model Definition ----
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(16, 4)  # 4 output classes

    def forward(self, x):
        out = self.encoder(x)  # (seq_len, batch, d_model)
        return self.fc(out.mean(dim=0))

def build_model():
    return TinyTransformer()

def build_example_input(model):
    return torch.randn(5, 1, 16)  # (seq_len, batch, d_model)


# ---- Training Script ----
def train():
    # Generate synthetic dataset (classification with 4 classes)
    num_samples = 1000
    X = torch.randn(5, num_samples, 16)   # (seq_len, batch, d_model)
    y = torch.randint(0, 4, (num_samples,))  # 4 classes

    dataset = TensorDataset(X.permute(1, 0, 2), y)  # make batch first
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    for epoch in range(1000):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # reshape back to (seq_len, batch, d_model)
            batch_x = batch_x.permute(1, 0, 2)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "./models/tiny_transformer.pth")
    print("✅ Model saved to tiny_transformer.pth")


if __name__ == "__main__":
    train()
