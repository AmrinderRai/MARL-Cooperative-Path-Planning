from pcb_twin.board import Board
from pcb_twin.connector import Connector
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from IPython.display import clear_output
import numpy as np
import random
import pdb

class MultiPathGridEnv():

    def __init__(self, obstacles, starts, goals, grid_size=[20,15], obs_size=[1000, 1000], agents_n=1, train=False, in_place=False): # **kwargs 
        self.train = train 
        self.in_place = in_place 
        self.partial_view = True if obs_size<grid_size else False 
        self.grid_size = grid_size
        self.obs_size = obs_size 
        self.obs_height = self.obs_size[0]
        self.obs_width = self.obs_size[1]
        self.height = self.grid_size[0]
        self.width = self.grid_size[1]
        self.obstacles = obstacles 
        self.connectors = []
        self.n_agents = agents_n
        self.action_space = list(range(0,8)) 
        # Assert that both these list sizes == n_agents after: 
        self.starts = starts
        self.goals = goals 
        self.episode_ended = [False]*self.n_agents 
        self._init_board()

    def _init_board(self):
        self.board = Board(board_size=self.grid_size) 
        self.to_connect = list(range(1,self.n_agents))
        self.connected = [] 

        if self.train:
            #obstacles_n = int((self.height/4) * (self.width/random.randint(5,6))) # <<-- Arbitrary: How many 
            obstacles_n = 2
            coord_set = set([(i,j) for i in range(self.height) for j in range(self.width)])
            self.obstacles = random.sample(coord_set, obstacles_n) 
            coord_set = coord_set.difference(self.obstacles)
            self.goals =  self.goals if self.in_place else random.sample(coord_set, self.n_agents) 
            coord_set = coord_set.difference(self.goals) 
            self.starts = self.starts if self.in_place else random.sample(coord_set, self.n_agents)

        for obstacle in self.obstacles: 
            self.board.place_obstacle(obstacle) 
        for i in range(0,self.n_agents): 
            i_d = i+1 
            connector = Connector(self.starts[i], i_d) 
            self.connectors.append(connector)
            self.board.cover_cell(self.starts[i], connector.head) 
            self.board.cover_cell(self.goals[i],  connector.goal)  

    def step(self, moves, idx): 
        # Assert agent == moves 
        rewards = []
        all_obs = [] 
        for i, move in zip(idx, moves):
            reward, obs = self._move_connector(i, move) 
            rewards.append(reward)
            all_obs.append(obs)

        return all_obs, rewards, self.episode_ended

    def start_obs(self): 
        # Assert agent == moves 
        all_obs = [] # TO DO
        for idx in range(self.n_agents):
            connector = self.connectors[idx] 
            head_coord = connector.get_head()
            dist = self._get_euclid_distance(head_coord, self.goals[idx])
            obs = (self.return_obs(head_coord), dist)
            all_obs.append(obs)

        return all_obs

    def reset(self):
        self.connectors = []
        self._init_board()
        self.episode_ended = [False]*self.n_agents
        return self.start_obs()

    def render(self):
        plt.ion()
        plt.figure(figsize=(20,10)) 
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        env_plot = self.board.grid.copy()
        env_plot = env_plot.astype(int)
        colors = ['grey', 'white', 'pink', 'red', 'green', 'blue', 'yellow', 'purple', 'purple']
        cmap = ListedColormap(colors[:self.n_agents+2])
        ax.matshow(env_plot, cmap=cmap) 

        plt.show()


    def _move_connector(self, idx, direction):
        connector = self.connectors[idx] 
        prev_pos = connector.get_head()
        self.board.grid[prev_pos] = connector.trail
        reward = -1 

        connector.move(direction)
        new_pos = connector.get_head()
        new_dist = self._get_euclid_distance(new_pos, self.goals[idx])

        if self.board.off_grid(new_pos): 
            reward -=10
            self.episode_ended = [True]*self.n_agents
            #connector.pop_head() useful if we don't want to end the episode.
        elif self.board.cell_val(new_pos)==connector.goal:
            self.board.cover_cell(new_pos,connector.head)
            self.episode_ended[idx] = True 
            reward += 20 
        elif self.board.cell_val(new_pos)!=self.board.EMPTY_CELL: 
            reward -=10  
            self.episode_ended = [True]*self.n_agents 
        else:
            self.board.cover_cell(new_pos,connector.head)
            if direction%2==1 and self.board.diag_collision(prev_pos, new_pos, direction):
                reward -=10
                self.episode_ended[idx] = True    

        
        return_obs = self.return_obs(connector.get_head())

        return reward, (return_obs, new_dist) 

    def return_obs(self, head):
        obs = self.partial_obs(head) if self.partial_view else self.board.grid
        obs_shaped = np.expand_dims(obs*10, axis=0) 
        return obs_shaped

    def partial_obs(self, head): 
        # Get bounding indices: 
        top_loc = head[0]-self.obs_height//2
        bottom_loc = head[0] + 1+ self.obs_height //2 
        left_loc = head[1] - self.obs_width //2
        right_loc = head[1] + 1 + self.obs_width //2

        if top_loc < 0:
            top = 0
            top_obs = np.abs(top_loc) 
        else:
            top = top_loc
            top_obs = 0

        if bottom_loc > self.height:
            bottom = self.height 
            bottom_obs = self.obs_height - np.abs(bottom_loc-self.height)
        else:
            bottom = bottom_loc
            bottom_obs = self.obs_height 

        if left_loc < 0: 
            left = 0 
            left_obs = np.abs(left_loc)
        else:
            left = left_loc
            left_obs = 0

        if right_loc > self.width:
            right = self.width
            right_obs = self.obs_width - np.abs(right_loc-self.width)
        else:
            right = right_loc
            right_obs = self.obs_width

        obs = self.board.grid[top:bottom, left:right] 

        padded_obs = np.ones((self.obs_height, self.obs_width))*[-1]

        padded_obs[top_obs:bottom_obs, left_obs:right_obs] = obs 

        return padded_obs

    def _get_euclid_distance(self, head, goal):
        head = np.array(head)
        goal = np.array(goal)
        vector_dist = head-goal 
        euclid_dist = np.sqrt(vector_dist[0]**2 + vector_dist[1]**2)

        return euclid_dist






