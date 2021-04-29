import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from torch.autograd import Variable
from itertools import zip_longest
import pdb

class RQNet(nn.Module):
    def __init__(self, action_dim):
        super(RQNet, self).__init__()
        self.number_of_actions = action_dim
        
        self.conv_feats = nn.Sequential(
            torch.nn.Conv2d(1, 5, 3, stride=2, padding=1) ,
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=81, hidden_size=256, num_layers=3)
        
        self.fc1 = nn.Linear(256, 856)  
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(856, self.number_of_actions)

    def forward(self, state, hidden, episode_lengths):
        grid_state, dist_state = state

        grid_state_reshaped = grid_state.reshape(-1, 1, 7, 7) 


        out = self.conv_feats(grid_state_reshaped)

        out = out.view(out.size(0), -1) 

        dist_state = dist_state.view(out.size(0), -1)

        out = torch.cat((out, dist_state), dim=1) 

        out = out.reshape(grid_state.size()[0], grid_state.size(1), -1)
        

        out = torch.nn.utils.rnn.pack_padded_sequence(out, episode_lengths.long(), batch_first=True, enforce_sorted=False) 

        out, hidden = self.lstm(out, hidden)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        
        return out, hidden

class DRQNAgent(object):
    def __init__(self, action_space, memory, device):
        self.device = device
        self.action_space = action_space
        self.qnet = RQNet(self.action_space).to(device) 
        self.target_net = RQNet(self.action_space).to(device) 
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(), lr=0.001) 
        self.discount_factor = 0.99 
        self.MSELoss_function = nn.MSELoss(reduction='none').to(device)  
        self.memory = memory
        self.reset_hidden()
        self.network_loss_history = []
        
    def epsilon_greedy_action(self, state, epsilon):
        network_output, hidden = self.qnet(state, self.hidden, torch.Tensor([1]).to(self.device))
        self.hidden = hidden 
        if np.random.uniform(0, 1) < epsilon: 
            return random.sample([i for i in range(self.action_space)],1)[0] 
        else: 
            network_output_to_numpy = network_output.cpu().data.numpy()[0]
            self.hidden = hidden 
            max_value = np.max(network_output_to_numpy)
            max_indices = np.nonzero(network_output_to_numpy == max_value)[1]
            policy_action = np.random.choice(max_indices) 
            return policy_action
    
    def policy_action(self, state):
        network_output, hidden = self.qnet(state, self.hidden, torch.Tensor([1]))
        self.hidden = hidden 
        network_output_to_numpy = network_output.cpu().data.numpy()[0]
        max_value = np.max(network_output_to_numpy)
        max_indices = np.nonzero(network_output_to_numpy == max_value)[1]
        policy_action = np.random.choice(max_indices) 
        return policy_action  
    
    def update_Q_Network(self, temporal_states, next_states, temporal_actions, temporal_rewards, hidden, episode_lengths):
        qs, _ = self.qnet(temporal_states, hidden, episode_lengths)
        temporal_actions = temporal_actions.unsqueeze(2)
        qsa = torch.gather(qs, dim=2, index=temporal_actions.long()) 
        
        qsa_mask = torch.zeros(qsa.size()[0:2]).to(self.device) 
        qsa_next_mask = torch.zeros(qsa.size()[0:2]).to(self.device)
        for batch_no, seq_length in enumerate(episode_lengths):
            non_terminals = seq_length-1
            qsa_mask[batch_no,0:seq_length] = torch.ones(seq_length)
            qsa_next_mask[batch_no,0:non_terminals] = torch.ones(non_terminals) 

        qsa *= qsa_mask.unsqueeze(2)

        qsa_next_actions, _ = self.target_net(next_states, hidden, episode_lengths)
        qsa_next_action, _ = torch.max(qsa_next_actions, dim=2, keepdim=True)
        temporal_rewards = temporal_rewards.unsqueeze(2) 
        
        qsa_next_target = temporal_rewards + qsa_next_mask.unsqueeze(2) * self.discount_factor * qsa_next_action
        qsa_next_target.to(self.device)
        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        q_network_loss = torch.mean(q_network_loss)
        self.network_loss_history.append(q_network_loss.cpu().data.numpy())
        self.qnet_optim.zero_grad() 
        q_network_loss.backward() 
        self.qnet_optim.step() 
        
    def update(self, update_rate, batch_size):
        for i in range(update_rate): 
            sample_batch = self.memory.sample_minibatch(batch_size)
            batch_size = len(sample_batch)
            episode_lengths = [len(episode) for episode in sample_batch]
            max_length = max(episode_lengths)
            temporal_states = np.zeros((batch_size, max_length), dtype=object)
            temporal_next = np.zeros((batch_size, max_length), dtype=object)
            temporal_actions = np.ones((batch_size, max_length))*[0] 
            temporal_rewards = np.ones((batch_size, max_length))* [0]
            for i, episode_length in enumerate(episode_lengths):
                sequence = np.array(sample_batch[i]) 
                for j, experience in enumerate(sequence[:episode_length]): 
                    temporal_states[i, j] = experience[0] 
                    temporal_next[i, j] = experience [1]
                    temporal_actions[i, j] = experience[2][0] 
                    temporal_rewards[i, j] = experience[3][0] 

            temporal_states = self.get_padded_states(temporal_states) 
            temporal_next = self.get_padded_states(temporal_next)
            temporal_actions = torch.Tensor(temporal_actions.tolist()).to(self.device)
            temporal_rewards = torch.Tensor(temporal_rewards.tolist()).to(self.device)
            episode_lengths = torch.LongTensor(episode_lengths).to(self.device)

            hidden = (Variable(torch.zeros(3, batch_size, 256).float().to(self.device)), Variable(torch.zeros(3, batch_size, 256).float().to(self.device)))

            self.update_Q_Network(temporal_states, temporal_next, temporal_actions, temporal_rewards, hidden, episode_lengths)

    def get_padded_states(self, state_np):
        batch_shape = state_np.shape 
        grid_states = np.zeros((batch_shape), dtype=object)
        dist_states = np.zeros((batch_shape), dtype=object)
        for i, seq in enumerate(state_np):
            for j, ep in enumerate(seq): 
                if type(state_np[i,j]) is not int:
                    grid_states[i,j] = ep[0] 
                    dist_states[i,j] = [ep[1]]
                else:
                    grid_states[i,j] = np.zeros((1,7,7)) 
                    dist_states[i,j] = [0]
        t_states = (torch.Tensor(grid_states.tolist()).to(self.device), torch.Tensor(dist_states.tolist()).to(self.device))

        return t_states

    def reset_hidden(self):
        self.hidden = (Variable(torch.zeros(3, 1, 256).float().to(self.device)), Variable(torch.zeros(3, 1, 256).float().to(self.device)))
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.qnet.state_dict())







