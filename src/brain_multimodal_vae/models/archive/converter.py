import torch.nn as nn
from torch.nn import functional as F

class Converter1(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, y_dim)
    
    def forward(self, x):
        return self.fc1(x)

class Converter2(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, y_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))

        return self.fc2(h)

class Converter3(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, y_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        return self.fc3(h)

class Converter4(nn.Module):
    def __init__(self, x_dim, y_dim, zp_dim, zs_dim, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, (zp_dim + zs_dim)*2)
        self.fc3 = nn.Linear((zp_dim + zs_dim)*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, y_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        return self.fc4(h)