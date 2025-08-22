import torch
import torch.nn as nn

class TinyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 4)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # last timestep

def build_model():
    return TinyRNN()

def build_example_input(model):
    return torch.randn(1, 5, 8)  # (batch, seq_len, input_size)
