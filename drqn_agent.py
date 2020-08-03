import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from torch.autograd import Variable

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def add_to_buffer(self, experience):
        self.buffer.append(experience)
        
    def sample_minibatch(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size)
        buffer_list = list(self.buffer)
        samples = [buffer_list[i] for i in sample_indices]
        return samples # <<<--- ** TO DO: LOOP OVER EPISODS AND THEN ZIP THEM SEPERATELY!!!! 

# TO DO: Generalise Q-net for diff grid dims (or patial obs): 
class RQNet(nn.Module):
    def __init__(self, action_dim):
        super(RQNet, self).__init__()
        self.number_of_actions = action_dim
        
        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=3, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1) 
        self.relu2 = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(input_size=181, hidden_size=181, num_layers=3)
        
        self.fc1 = nn.Linear(181, 856)  
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(856, self.number_of_actions)

    def forward(self, state, context):
        grid_state, dist_state = state

        out = self.conv1(grid_state)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out.view(out.size(0), -1) # Flatten

        out = torch.cat((out, torch.unsqueeze(dist_state,1)), dim=1) 
        
        # TO DO: Change fully connected MLP to LSTM (DRQN)... 
        out = out.unsqueeze(1) # WHY???!!
        out, context = self.lstm(out, context)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        
        return out, context

class DRQNAgent(object):
    def __init__(self, action_space, device):
        self.device = device
        self.action_space = action_space
        self.qnet = RQNet(self.action_space).to(device) 
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(), lr=0.001) 
        self.discount_factor = 0.99 
        self.MSELoss_function = nn.MSELoss().to(device)  
        self.memory = ReplayBuffer(1000)
        #self.network_loss_history = []
        
    def epsilon_greedy_action(self, state, context, epsilon):
        if np.random.uniform(0, 1) < epsilon: 
            return random.sample([i for i in range(self.action_space)],1)[0], context 
        else: 
            network_output_to_numpy, context = self.qnet(state, context)
            network_output_to_numpy = network_output_to_numpy.cpu().data.numpy()
            max_value = np.max(network_output_to_numpy)
            max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
            policy_action = np.random.choice(max_indices) 
            return policy_action, context  
    
    def policy_action(self, state):
        network_output_to_numpy = self.qnet(state).cpu().data.numpy()
        max_value = np.max(network_output_to_numpy)
        max_indices = np.nonzero(network_output_to_numpy == max_value)[0]
        policy_action = np.random.choice(max_indices) 
        return policy_action  
    
    def update_Q_Network(self, state, action, reward):
        # THIS UPDATE WILL BE TOTALLY DIFFERENT... Context/Hidden?! where will that come from... 
        context = (Variable(torch.zeros(3, 1, 181).float()), Variable(torch.zeros(3, 1, 181).float()))
        qsa = torch.gather(self.qnet(state, context), dim=1, index=action.long())
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
            sample_batch = self.memory.sample_minibatch(20)
            
            # NOTE: This for loop can be removed... 
            # ... if we do vectorised batch update (will need padding for diff episode lengths)
            for i, sample in enumerate(sample_batch): 
                states, actions, rewards = map(list, zip(*sample)) 
                states = tuple(map(torch.Tensor, zip(*states)))
                states = (states[0].to(self.device),states[1].to(self.device))
                actions = torch.Tensor(actions).to(self.device) # May need to give it a good squeeze??
                rewards = torch.Tensor(rewards).to(self.device)
                self.update_Q_Network(states, actions, rewards)



