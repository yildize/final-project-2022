import numpy as np
import torch
import torch.nn as nn
import itertools
from network_ppo import DenseNet
from utils import Memory_PPO
from torch.autograd import Variable
from torch.distributions import MultivariateNormal


class Model(torch.nn.Module):
    def __init__(self, env, params, n_insize, n_outsize, initial_act_std=0.6):
        super().__init__()
        self.n_states = n_insize
        self.n_actions = n_outsize
        self.gamma = params.gamma
        self.n_epochs = params.n_epochs
        self.value_coeff = params.value_coeff
        self.entropy_coeff = params.entropy_coeff
        self.episodes = params.episodes
        self.clip_range = params.clip_range
        self.action_std = initial_act_std
        

        #create network:
        self.policy = DenseNet(self.n_states, self.n_actions, initial_act_std)
        self.optim = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': params.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': params.lr_critic}
        ])
        
        
        self.policy_old = DenseNet(self.n_states, self.n_actions, initial_act_std)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        # make a replay buffer memory ro store experiences
        self.memory = Memory_PPO(params.rollout_len)


        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
            
            
    def select_action(self, state):  
        
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)

        return action.detach().cpu().numpy()[0], action_logprob
        
        
    def forward_given_actions(self, state: torch.Tensor, action: torch.Tensor):

        log_probs, values, entropies = self.policy.evaluate(state, action)
        return log_probs, values, entropies
    

    def calculate_returns(self, memories, rollout_len):

        advantages_list = []
        
        Qval = 0 
        returns_list = [] 

        with torch.no_grad():
            for t in reversed(range(rollout_len)):
                #Return calculation
                Qval = memories.rewards[t] + self.gamma * Qval * memories.dones[t]
                returns_list.insert(0,Qval)
        
        return advantages_list, returns_list
        

    def update(self,  rollout_data_gen):

        #We'll be storing batch losses
        losses = []

        #clip_range = next(clip_schedule)
        #self.clip_range =  max(0.15, self.clip_range - 0.001)

        #For n_epochs:
        for _ in range(self.n_epochs):
            #Loop through batches:
            for batch_data in rollout_data_gen:

                #Obtain batch components:
                states, actions, rewards, old_log_probs, dones, returns = batch_data
                old_log_probs = old_log_probs.detach() #[bs,1]
                returns = returns.detach() #[bs,1]
                #Gradients false

                #Get new probs, values and entropies:
                new_log_probs, values, entropy = self.forward_given_actions(states,actions)
                #values = torch.squeeze(values)
                advantages = returns - values.detach()
        
                #Calculate clipped objective
                prob_ratios = torch.exp(new_log_probs - old_log_probs)
                weighted_probs = prob_ratios * advantages
                weighted_clipped_probs = torch.clamp(prob_ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages

                #Calculate loss terms:
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                critic_loss = self.MseLoss(values, returns)

                #Calculate final cost:
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                
                #Take a gradient step
                self.optim.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()
                
                #Record loss
                losses.append(loss.mean().detach())


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        print('************ ROLLOUT UPDATE PERFORMED *************** ')
        print('Loss:',sum(losses)/len(losses))
        print('Action std:', self.action_std)
        print()