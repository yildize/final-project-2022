import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

class DenseNet(torch.nn.Module):
    
    def __init__(self, in_size: int, out_size: int, initial_act_std, hidden: int = 128, ):
        super().__init__()
        self.out_size = out_size
        
        self.action_var = torch.full((out_size,), initial_act_std * initial_act_std)

        self.actor = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_size),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    
    def set_action_std(self, act_std):
        self.action_var = torch.full((self.out_size,), act_std * act_std)
        
    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        action_logprob = dist.log_prob(action)
    
        return action.detach(), action_logprob.detach()
    
    
    def evaluate(self, state, action):
    

        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
    

        return action_logprobs, state_values, dist_entropy