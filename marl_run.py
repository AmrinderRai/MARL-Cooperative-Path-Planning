from pcb_envs import MultiPathGridEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from agent_trainer import QLearner

n_agents = 1
grid_size = [15,10]
obstacles = [(3,3),(6,2), (6,3), (9,4), (9,3), (10,4), (8,9), (8,8)]
starts = [(13,8)]
goals = [(1,1)] # orig: (2,1) 
to_train = True 

env1 = MultiPathGridEnv(obstacles, starts, goals, grid_size=grid_size, agents_n=n_agents, train=to_train) 

action_dim = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn_agent = DQNAgent(action_dim, device)

qlearner = QLearner(dqn_agent, env1)

qlearner.train_agent()

qlearner.save_learning()


