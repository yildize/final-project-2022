import numpy as np
import random
from collections import deque, namedtuple
import torch


class Memory:
    def __init__(self, buffer_size):
        self.memories = deque(maxlen=buffer_size)

    # append experience to the replay memory
    def push(self, state, action, reward, next_state, done):
        self.memories.append((state, action, np.array([reward]), next_state, done))

    # get experience sample of batch size
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        # get one batch randomly from the replay memory
        batch = random.sample(self.memories, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memories)



class Memory_PPO:
    def __init__(self, rollout_len):
        self.Memory = namedtuple("Memory", "states actions rewards log_probs dones values")
        self.memories =  self.Memory([],[],[],[],[],[])
        self.rollout_len = rollout_len
        self.next_value = None

    # append experience to the replay memory
    def push(self, state, action, reward, next_state, log_prob, value, done):
        self.memories.states.append(torch.FloatTensor(state).unsqueeze(0)) #state, next state numpy array of (41,)
        self.memories.actions.append(torch.FloatTensor(action).unsqueeze(0)) #action numpy array of (2,)
        self.memories.rewards.append(torch.tensor([[reward]])) #reward int
        self.memories.log_probs.append(log_prob) #log prob tensor of size [1,1]
        self.memories.dones.append(torch.tensor([[1-done*1]])) #done boolean
        self.memories.values.append(value) #value tensor of size [1,1]
       
    

    # get experience sample of batch size
    def sample(self, n_epochs, batch_size, advantages, returns):

        states = torch.cat(self.memories.states) #[rollout_len, 41]
        actions = torch.cat(self.memories.actions) #[rollout_len, 2]
        rewards = torch.cat(self.memories.rewards) #[rollout_len, 1]
        log_probs = torch.cat(self.memories.log_probs).detach() #[rollout_len, 1]
        dones = torch.cat(self.memories.dones) #[rollout_len, 1]
        
        advantages = torch.cat(advantages, dim=0) #[rollout_len, 1]
        returns = torch.cat(returns, dim=0) #[rollout_len, 1]        

        for _ in range(n_epochs):
            indices = np.arange(self.rollout_len, dtype=np.int64)
            np.random.shuffle(indices)
            for start_index in np.arange(0, self.rollout_len, batch_size):
                batch_indices = indices[start_index:start_index+batch_size]
                #batch_indices = np.random.choice(indices, batch_size, replace=False)
                yield  states[batch_indices], actions[batch_indices], rewards[batch_indices], log_probs[batch_indices],  dones[batch_indices], advantages[batch_indices], returns[batch_indices], batch_indices

    def clean(self):
        self.memories =  self.Memory([],[],[],[],[],[])

    def __len__(self):
        return len(self.memories.rewards)
