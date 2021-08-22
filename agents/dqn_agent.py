import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def add_to_buffer(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
        
    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
        
    def sample_minibatch(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        buffer_list = list(self.buffer)
        samples = [buffer_list[i] for i in sample_indices]
        return map(list, zip(*samples)) 

# TO DO: Generalise Q-net for diff grid dims (or patial obs): 
class Q_ConvNet(nn.Module):
    def __init__(self, action_dim):
        super(Q_ConvNet, self).__init__()
        self.number_of_actions = action_dim
        
        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=3, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1) 
        self.relu2 = nn.ReLU(inplace=True)
        
        self.fc1 = nn.Linear(181, 856) 
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(856, self.number_of_actions)
        

    def forward(self, state):
        grid_state, dist_state = state

        out = self.conv1(grid_state)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out.view(out.size(0), -1) 

        out = torch.cat((out, torch.unsqueeze(dist_state,1)), dim=1) 
        
        # TO DO: Change fully connected MLP to LSTM (DRQN)... 
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        
        return out 

class DQNAgent(object):
    def __init__(self, action_space, device):
        self.device = device
        self.action_space = action_space
        self.qnet = Q_ConvNet(self.action_space).to(device) 
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(), lr=0.001) 
        self.discount_factor = 0.99 
        self.MSELoss_function = nn.MSELoss().to(device)  
        self.memory = PrioritizedReplayBuffer(1000)
        #self.network_loss_history = []
        
    def epsilon_greedy_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon: 
            return random.sample([i for i in range(self.action_space)],1)[0] 
        else: 
            network_output_to_numpy = self.qnet(state).cpu().data.numpy()
            max_value = np.max(network_output_to_numpy)
            max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
            policy_action = np.random.choice(max_indices) 
            return policy_action  
    
    def policy_action(self, state):
        network_output_to_numpy = self.qnet(state).cpu().data.numpy()
        max_value = np.max(network_output_to_numpy)
        max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
        policy_action = np.random.choice(max_indices) 
        return policy_action  
    
    def update_Q_Network(self, state, next_state, action, reward, terminal):
        qsa = torch.gather(self.qnet(state), dim=1, index=action.long())
        qsa_next_actions = self.qnet(next_state)
        qsa_next_action, _ = torch.max(qsa_next_actions, dim=1, keepdim=True)
        not_terminal = 1-terminal 
        reward = reward.resize_(not_terminal.size()) 
        qsa_next_target = reward + not_terminal * self.discount_factor * qsa_next_action
        qsa_next_target.to(self.device)
        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        #self.network_loss_history.append(q_network_loss.double()) <-- PROBLEM: GPU leakage! 
        self.qnet_optim.zero_grad() 
        q_network_loss.backward() 
        self.qnet_optim.step() 
        
    def update(self, update_rate):
        for i in range(update_rate):
            states, next_states, actions, rewards, terminals = self.memory.sample_minibatch(20)
            states = tuple(map(torch.Tensor, zip(*states)))
            states = (states[0].to(self.device),states[1].to(self.device))
            next_states = tuple(map(torch.Tensor, zip(*next_states)))
            next_states = (next_states[0].to(self.device),next_states[1].to(self.device))
            actions = torch.Tensor(actions).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            terminals = torch.Tensor(terminals).to(self.device)
            self.update_Q_Network(states, next_states, actions, rewards, terminals)



