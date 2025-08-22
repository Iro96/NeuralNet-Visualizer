import torch
import torch.nn as nn

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

def build_model():
    return TinyAutoencoder()

def build_example_input(model):
    return torch.randn(1, 20)
