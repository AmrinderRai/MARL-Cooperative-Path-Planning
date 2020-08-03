import torch 
import csv
import time 
import sys 

class QLearner():
    def __init__(self, agent, env):
        self.agent = agent 
        self.reward_history = []
        self.td_error_history = [] # <<-- We'll leave exporting this for now
        self.epsilon_history = []
        self.env = env 

    def _reset_history(self):
        self.reward_history = []
        self.td_error_history = []        

    def train_agent(self, epsilon=0.6, decay=0.9999, epochs=1000000, grid_size=[15,10]):
        self._reset_history()
        max_time_steps = 3000 
        min_epsilon = 0.1
        for episode in range(epochs):
            reward_sum = 0
            state = self.env.reset()
            epsilon = max(min_epsilon, epsilon*decay)
            self.epsilon_history.append(epsilon)
            for i in range(max_time_steps):
                action = self.agent.epsilon_greedy_action(torch.from_numpy(state).reshape(1,1,grid_size[0],grid_size[1]).float(), epsilon)
                if type(action) is not list:
                    action = [action]
                next_state, reward, terminal = self.env.step(action)
                reward_sum += reward[0]
                self.agent.memory.add_to_buffer((state.reshape(1,grid_size[0],grid_size[1]), next_state.reshape(1,grid_size[0],grid_size[1]), action, [reward], [terminal]))
                state = next_state
                #env1.render()
                if terminal:
                    print('Current episode: ' + str(episode) + '\r')
                    self.reward_history.append(reward_sum)
                    reward_sum = 0
                    break
            if episode !=0:
                self.agent.update(40)

    def save_learning(self, dir = '/'):
        file_name = 'learning_data.csv'
        wtr = csv.writer(open (file_name, 'w'), delimiter=',', lineterminator='\n')
        for x in self.reward_history : wtr.writerow ([x]) 
        #for x in self.epsilon_history : wtr.writerow ([x]) TO DO: Also export epsilon history into its own column!! 

    def save_params(self, dir = "/"):
        # TO DO: Save PyTorch model params to dir (default root dir)
        return None 
