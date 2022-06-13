import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal

class DenseNet(torch.nn.Module):
    
    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.ReLU()
        )

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(hidden, out_size),
            torch.nn.Tanh()
        )

        self.std = torch.nn.Sequential(
            torch.nn.Linear(hidden, out_size),
            torch.nn.Softplus()
        )

        self.value = torch.nn.Linear(hidden,1)

    def forward(self, state: torch.Tensor):
        x = self.base(state)
        mu = self.mu(x)
        std = torch.clamp(self.std(x),0,3)
        dist = Normal(mu, std)
        #cov_mat = torch.diag_embed(std)
        #dist = MultivariateNormal(mu, cov_mat)
        
        return dist, self.value(x)