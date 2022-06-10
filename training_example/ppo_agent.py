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
        self.tau = params.tau
        self.alpha = params.alpha
        self.hidden_size = params.hidden

        #create network:
        network = DenseNet(self.n_states, self.n_actions, self.hidden_size)
        
        #optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=params.lr) #LR SHOULD BE ADDED TO THE PARAMS IN TRAIN MODULE!


        # make a replay buffer memory ro store experiences
        self.memory = Memory_PPO(params.buffersize)

        # mse loss algoritm is applied
        self.critic_criterion = torch.nn.MSELoss()

        # define actor and critic network optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.lrpolicy)
        self.critic_optimizer = torch.optim.Adam(self.q_params, lr=params.lrvalue)
        
        
    def select_action(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        #First obtain actor and critic outputs: (network will return categorical dists directly instead of logits)
        dists, values = self.network(state) #don't forget that states obtained from vecenv which can be taught as batch of environments
        #[n_envs, n_acts]

        #Calculate actions and their corresponding log_probs:
        actions = dists.sample()#[n_envs,1]

        torch.clamp(actions, min=-1, max=1)
        log_probs = dists.log_prob(actions).sum(1).unsqueeze(1) #[n_envs,1]

        return actions.detach().cpu().numpy()[0], log_probs,  values
        #     [n_envs,1]  [n_envs,1] [n_envs,1]
        
        
    def forward_given_actions(self, state: torch.Tensor, action: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dists, values = self.network(state)

        log_probs = dists.log_prob(action).sum(1).unsqueeze(1) #[n_envs,1]
        entropies = dists.entropy().sum(1).unsqueeze(1) #[n_envs,1]

        return log_probs, values, entropies
    
    
    def collect_rollout(self, states: np.ndarray, ) -> Tuple[Rollout, np.ndarray]:
        rollout_list = []
        for step in range(self.args.n_step):

            # Turn states into torch tensor:
            states_t = torch.FloatTensor(states, device=self.args.device)  # (venv,n_state)

            actions, log_probs, values = self.forward(states_t)
            # [n_envs,1],[n_envs,1] [n_envs,1] [n_envs,1],[n_envs, hidden_size]

            actions_np = actions.numpy()

            # Take that action for each environment
            next_states, rewards, dones = self.vecenv.step(actions_np)
            # np arrays of (n_envs,n_states)  (n_envs,1), (n_envs,1)

            # Record transition
            trans = self.Transition(torch.Tensor(rewards), torch.Tensor(1.0 - dones), states_t, actions, log_probs,values)
            rollout_list.append(trans)

            states = next_states

        _, next_values = self.network(torch.tensor(next_states, dtype=torch.float32, device=self.args.device))

        return (self.Rollout(rollout_list, next_values.detach()), next_states)
    
    

    def calculate_gae(rollout: Rollout, gamma: float, gae_lambda: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:


        advantages = torch.zeros(rollout.target_value.size()[0] ,len(rollout.list)+1)
        advantages_list = [None]*len(rollout.list)

        Qval = rollout.target_value
        returns_list = []  # [None]*len(rollout.list)
        
        rollout_list = rollout.list
        with torch.no_grad():
            for t in reversed(range(len(rollout.list))):

                #ADVANTAGE CALCULATION
                #If that is the last rollout timestep:
                if t == len(rollout.list) - 1:
                    next_val = rollout.target_value
                else:
                    next_val = rollout_list[t+1].value

                delta = rollout_list[t].reward + (gamma * next_val * rollout_list[t].done) - rollout_list[t].value
                advantages[:,t] = torch.add(delta.squeeze(), (gamma * gae_lambda * advantages[:,t+1].unsqueeze(1) * (rollout_list[t].done)).squeeze())
                advantages_list[t] = advantages[:,t].unsqueeze(1)

                #RETURN CALCULATION
                Qval = rollout_list[t].reward + gamma * Qval * rollout_list[t].done
                returns_list.insert(0,Qval)

        return advantages_list, returns_list
    
    
    def rollout_data_loader(rollout: Rollout, advantages: List[torch.Tensor], returns: List[torch.Tensor],) -> TrainData:
        
        #reward done state action log_prob value entropy

        # [16*5,1],[16*5,1], [16*5, 8], [16*5, 1]
        rewards, dones, states, actions, log_probs, values, entropies = [
            torch.cat(tensor, dim=0) for tensor in zip(*rollout.list)
        ]
        advantages = torch.cat(advantages, dim=0)
        returns = torch.cat(returns, dim=0)
        return A2C.TrainData(
            log_probs, #[16*5, 1]
            advantages, #[16*5, 1]
            returns, #[16*5, 1]
            values, #[16*5, 1]
            entropies, #[16*5, 1]
        )
        
        
    def update(self,  rollout_data: TrainData) -> Tuple[float, float, float]:
        # Dont forget to detach returns and advantages
        
        returns = rollout_data.returns.detach()
        log_probs = rollout_data.log_prob
        values = rollout_data.value
        entropies = rollout_data.entropy

        gae = rollout_data.advantage.detach()
        advantages = returns - values

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(log_probs * gae).mean()
        entropy_loss = entropies.mean()

        loss = actor_loss + self.args.value_coef * critic_loss - self.args.entropy_coef * entropy_loss

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) #0.5
        self.optim.step()