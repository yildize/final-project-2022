import numpy as np
import torch
import itertools
from network_ppo import DenseNet
from utils import Memory_PPO
from torch.autograd import Variable


class Model(torch.nn.Module):
    def __init__(self, env, params, n_insize, n_outsize):
        super().__init__()
        self.n_states = n_insize
        self.n_actions = n_outsize
        self.gamma = params.gamma
        self.n_epochs = params.n_epochs
        self.value_coeff = params.value_coeff
        self.entropy_coeff = params.entropy_coeff
        self.episodes = params.episodes
        self.clip_range = 0.1

        #create network:
        self.network = DenseNet(self.n_states, self.n_actions)
        
        #optimizer
        self.optim = torch.optim.Adam(self.network.parameters(), lr=params.lr)

        # make a replay buffer memory ro store experiences
        self.memory = Memory_PPO(params.rollout_len)

        
    def select_action(self, state):
        
        state = torch.FloatTensor(state).unsqueeze(0) #[1,41]
        dists, value = self.network(state) # value [1,1]

        #Calculate actions and their corresponding log_probs:
        actions = dists.sample() #[1,2]
        actions = torch.clamp(actions, min=-1, max=1) #[1,2]
        log_prob = dists.log_prob(actions).sum(1).unsqueeze(0) #[1,1]
        
        return actions.detach().cpu().numpy()[0], log_prob,  value
        
        
    def forward_given_actions(self, state: torch.Tensor, action: torch.Tensor):

        dists, values = self.network(state)

        log_probs = dists.log_prob(action).sum(1).unsqueeze(1) #[n_envs,1]
        entropies = dists.entropy().sum(1).unsqueeze(1) #[n_envs,1]

        return log_probs, values, entropies
    

    def calculate_returns(self, memories, rollout_len, next_val):

        #Initialize the advantages:
        advantages = torch.zeros(next_val.size()[0] ,rollout_len+1)
        advantages_list = [None]*rollout_len
        gae_lambda = 0.95

        Qval = next_val
        returns_list = [] 

        with torch.no_grad():
            for t in reversed(range(rollout_len)):

                #Advantage Calculation
                #If that is the last rollout timestep:
                if t == rollout_len - 1:
                    next_v = next_val
                else:
                    next_v = memories.values[t+1]
                
                #First calculate the td error and then gae:
                delta = memories.rewards[t] + (self.gamma * next_v * memories.dones[t]) - memories.values[t]
                advantages[:,t] = torch.add(delta.squeeze(), (self.gamma * gae_lambda * advantages[:,t+1].unsqueeze(1) * (memories.dones[t])).squeeze())
                advantages_list[t] = advantages[:,t].unsqueeze(1)

                #Return calculation
                Qval = memories.rewards[t] + self.gamma * Qval * memories.dones[t]
                returns_list.insert(0,Qval)
        
        return advantages_list, returns_list
        

    def update(self,  rollout_data_gen):

        #We'll be storing batch losses
        critic_losses = []
        actor_losses = []
        entropy_losses = []

        #clip_range = next(clip_schedule)
        #self.clip_range =  max(0.15, self.clip_range - 0.001)

        #For n_epochs:
        for _ in range(self.n_epochs):
            #Loop through batches:
            for batch_data in rollout_data_gen:
               
                #Gradients false
                #Obtain batch components:
                states, actions, rewards, old_log_probs, dones, advantages, returns, batch_indices = batch_data
                #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10) #Further reduce the variance!
                
                #Gradients true
                #Get new probs, values and entropies:
                new_log_probs, values, entropies = self.forward_given_actions(states,actions)
                #[bs,1], [bs,1], [bs,1]  with gradients True
                #advantages = returns - values.detach() 
                
                #advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-5)
                #returns = (returns - returns.mean()) / max(returns.std(), 1e-5)

                #Calculate clipped objective
                prob_ratios = new_log_probs.exp()/old_log_probs.exp()
                weighted_probs = prob_ratios * advantages
                weighted_clipped_probs = torch.clamp(prob_ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages

                #Calculate loss terms:
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                critic_loss = (torch.sqrt((returns-values).pow(2))).mean()
                entropy_loss = entropies.mean()

                #Calculate final cost:
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy_loss

                #Take a gradient step
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optim.step()

                #Store batch losses
                critic_losses.append(critic_loss.detach().item())
                actor_losses.append(actor_loss.detach().item())
                entropy_losses.append(entropy_loss.detach().item())

        print()
        print('************ ROLLOUT UPDATE PERFORMED *************** ')
        print('Critic Loss:',sum(critic_losses)/len(critic_losses), 
              ' Actor Loss:', sum(actor_losses)/len(actor_losses),
              ' Entropy Loss:', sum(entropy_losses)/len(entropy_losses))
        print()