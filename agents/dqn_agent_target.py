import numpy as np
import torch
import torch.nn as nn
import random

class Q_ConvNet(nn.Module):
    def __init__(self, action_dim):
        super(Q_ConvNet, self).__init__()
        """ For a 7x7 partial obs view"""
        self.number_of_actions = action_dim
        
        self.conv1 = torch.nn.Conv2d(1, 5, 3, stride=2, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)
        
        self.fc1 = nn.Linear(81, 286) 
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(286, self.number_of_actions)
        

    def forward(self, state):
        grid_state, dist_state = state

        out = self.conv1(grid_state)
        out = self.relu1(out)
        out = out.view(out.size(0), -1) 

        out = torch.cat((out, torch.unsqueeze(dist_state,1)), dim=1) 
        
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        
        return out 
        

class DQNAgent(object):
    def __init__(self, action_space, memory, device):
        self.device = device
        self.action_space = action_space
        self.qnet = Q_ConvNet(self.action_space).to(device) 
        self.target_net = Q_ConvNet(self.action_space).to(device) 
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(), lr=0.001) 
        self.discount_factor = 0.99 
        self.MSELoss_function = nn.MSELoss(reduction='none').to(device)  
        self.memory = memory
        self.network_loss_history = []
        
    def epsilon_greedy_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon: 
            return random.sample([i for i in range(self.action_space)],1)[0] 
        else: 
            network_output_to_numpy = self.qnet(state).cpu().data.numpy()[0]
            max_value = np.max(network_output_to_numpy)
            max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
            policy_action = np.random.choice(max_indices) 
            return policy_action  
    
    def policy_action(self, state):
        network_output_to_numpy = self.qnet(state).cpu().data.numpy()[0]
        max_value = np.max(network_output_to_numpy)
        max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
        policy_action = np.random.choice(max_indices) 
        return policy_action  
    
    def update_Q_Network(self, state, next_state, action, reward, terminal, importance, indices):
        qsa = torch.gather(self.qnet(state), dim=1, index=action.long())
        qsa_next_actions = self.target_net(next_state)
        qsa_next_action, _ = torch.max(qsa_next_actions, dim=1, keepdim=True)
        not_terminal = 1-terminal 
        reward = reward.resize_(not_terminal.size()) 
        qsa_next_target = reward + not_terminal * self.discount_factor * qsa_next_action
        qsa_next_target.to(self.device)
        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        errors = q_network_loss.squeeze(1).cpu().data.tolist()
        self.memory.set_priorities(indices, errors)
        q_network_loss = torch.mean(q_network_loss)
        self.network_loss_history.append(q_network_loss.cpu().data.numpy())
        self.qnet_optim.zero_grad() 
        q_network_loss.backward() 
        self.qnet_optim.step() 
        
    def update(self, update_rate, batch_size):
        for i in range(update_rate):
            (states, next_states, actions, rewards, terminals), importance, indices = self.memory.sample_minibatch(batch_size)
            states = tuple(map(torch.Tensor, zip(*states)))
            states = (states[0].to(self.device),states[1].to(self.device))
            next_states = tuple(map(torch.Tensor, zip(*next_states)))
            next_states = (next_states[0].to(self.device),next_states[1].to(self.device))
            actions = torch.Tensor(actions).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            terminals = torch.Tensor(terminals).to(self.device)
            self.update_Q_Network(states, next_states, actions, rewards, terminals, importance, indices)

    def update_target_net(self):
        self.target_net.load_state_dict(self.qnet.state_dict())



