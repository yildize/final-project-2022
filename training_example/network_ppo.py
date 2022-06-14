import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal, Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DenseNet(torch.nn.Module):

    
    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        super().__init__()
        out_size = 6
        
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(in_size, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 1), std=1.0),
        )
            
        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(in_size, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, out_size), std=0.01),
        )

    def forward(self, state: torch.Tensor):        
        return Categorical(logits=self.actor(state)), self.critic(state)