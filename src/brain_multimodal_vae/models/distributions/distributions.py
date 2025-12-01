import torch
import torch.nn as nn
from torch.nn import functional as F
from pixyz.distributions import Normal, Laplace
from pixyz.utils import epsilon

class Inference(Normal):
    def __init__(self, enc, var, cond_var, z_dim, hidden_dim=1024):
        super(Inference, self).__init__(var=var, cond_var=cond_var, name="q")

        self.enc = enc
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, **x):
        x = torch.cat([x[_cond_var] for _cond_var in self.cond_var], dim=1)
        h = self.enc(x)

        return {"loc": self.mu(h), "scale": F.softplus(self.logvar(h)) + epsilon()}

class Generator(Laplace):
    def __init__(self, dec, var, cond_var, z_dim, hidden_dim=1024):
        super(Generator, self).__init__(var=var, cond_var=cond_var, name="p")

        self.fc = nn.Linear(z_dim, hidden_dim)
        self.dec = dec

    def forward(self, **z):
        z = torch.cat([z[_cond_var] for _cond_var in self.cond_var], dim=1)
        h = F.relu(self.fc(z))

        return {"loc": self.dec(h), "scale": torch.tensor(1.0).to(z.device)}