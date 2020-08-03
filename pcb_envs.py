from components.board import Board
from components.connector import Connector
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from IPython.display import clear_output
import numpy as np
import random

class MultiPathGridEnv():

    def __init__(self, obstacles, starts, goals, grid_size=[20,15], obs_size=[7,7], agents_n=1, train=False): # <<-- REFACTOR: Use **kwargs :) 
        self.train = train # Perhaps we can switch to not training later << :) 
        self.grid_size = grid_size
        self.obs_size = obs_size # << obs_diam might be easier??? ... (Assert odd dims?!)
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
        self.episode_ended = False 
        self._init_board()

    def _init_board(self):
        self.board = Board(board_size=self.grid_size) 
        self.to_connect = list(range(1,self.n_agents)) # TO DO: Remove connected idx from here.. 
        self.connected = [] # ... add to here (or )

        if self.train:
            obstacles_n = int((self.height/4) * (self.width/random.randint(5,6))) # <<-- Arbitrary: How many 
            coord_set = set([(i,j) for i in range(self.height) for j in range(self.width)])
            self.obstacles = random.sample(coord_set, obstacles_n) 
            coord_set = coord_set.difference(self.obstacles)
            self.goals = random.sample(coord_set, self.n_agents)
            coord_set = coord_set.difference(self.goals)
            self.starts = random.sample(coord_set, self.n_agents) 

        for obstacle in self.obstacles: 
            self.board.place_obstacle(obstacle) 
        for i in range(0,self.n_agents): 
            i_d = i+1 
            connector = Connector(self.starts[i], i_d) 
            self.connectors.append(connector)
            self.board.cover_cell(self.starts[i], connector.head) 
            self.board.cover_cell(self.goals[i],  connector.goal)  

    def step(self, moves): 
        # Assert agent == moves 
        rewards = []
        all_obs = [] # TO DO
        for i, move in enumerate(moves):
             # TO DO: get idx from.... idx = self.to_connect[i]  
            reward, obs = self._move_connector(i, move) # TO DO: get partial obs to into a list. 
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
            obs = (self.partial_obs(head_coord)*10, dist)
            all_obs.append(obs)

        return all_obs

    def reset(self):
        self.connectors = []
        self._init_board()
        self.episode_ended = False
        return self.start_obs()

    def render(self):
        # Turn interactive mode on.
        plt.ion()
        plt.figure(figsize=(20,10)) 
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment plot
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

        # Move and adjust reward based on euclid distance change... 
        prev_dist = self._get_euclid_distance(prev_pos, self.goals[idx])
        connector.move(direction)
        new_pos = connector.get_head()
        new_dist = self._get_euclid_distance(new_pos, self.goals[idx])
        dist_change = prev_dist - new_dist
        reward += dist_change

        if self.board.off_grid(new_pos): 
            #print("Off the Grid !!!")
            reward -=10
            self.episode_ended = True
            #connector.pop_head() <<-- Only useful if we don't want to end the episode. :)
        elif self.board.cell_val(new_pos)==connector.goal:
            #print("Reached goal!!!")
            self.board.cover_cell(new_pos,connector.head)
            self.episode_ended = True # << TO DO!!!: Modify for MARL ... Only that connector is done!!! Check out how the Gym-Snake guy did it :)
            reward += 20 
        elif self.board.cell_val(new_pos)!=self.board.EMPTY_CELL: # Collision with obstacle (including other connectors)
            #print("Collision occured !!!")
            reward -=10  
            self.episode_ended = True 
        else:
            self.board.cover_cell(new_pos,connector.head)
            if direction%2==1 and self.board.diag_collision(prev_pos, new_pos, direction):
                reward -=10
                self.episode_ended = True    

        # TO DO: get partial_obs then return along with the reward!! 

        return reward, (self.partial_obs(connector.get_head())*10, new_dist) # << NOTE: *10 is a quick fix for now :) 

    # BETTER TO IMPLEMENT THIS IN BOARD CLASS: 
    def partial_obs(self, head): 
        # Get bounding indices: 
        top = max(0, head[0]-self.obs_height//2)
        bottom = min(self.height,head[0] + 1+ self.obs_height //2) # Includes middle [assuming odd]
        left = max(0,head[1] - self.obs_width //2)
        right = min(self.width, head[1] + 1 + self.obs_width //2) # Includes middle [assuming odd]

        # TO DO: Which side to pad??? <<< concat lr, bl, b etc. to obs as tuple?? 
        # OR more convenient to pad here???? 

        obs = self.board.grid[top:bottom, left:right] 

        return obs 

    def _get_euclid_distance(self, head, goal):
        head = np.array(head)
        goal = np.array(goal)
        vector_dist = head-goal 
        euclid_dist = np.sqrt(vector_dist[0]**2 + vector_dist[1]**2)

        return euclid_dist

    def _end_connection(self, connector):
        # TO DO: # Remove connection from to_connect to connecting 
        return None 






