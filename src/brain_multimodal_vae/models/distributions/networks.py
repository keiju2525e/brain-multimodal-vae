import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024):
        super().__init__()
        self.fc = nn.Linear(x_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, x_dim)

    def forward(self, x):
        return self.fc(x)