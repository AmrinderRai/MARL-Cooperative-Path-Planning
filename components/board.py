import numpy as np

class Board(): 
    EMPTY_CELL  = 0
    OBSTACLE_CELL = -1

    def __init__(self, board_size=[30,20]): 
        self.height = board_size[0]
        self.width = board_size[1]
        self.grid = np.zeros((self.height, self.width),dtype=float)
        self.grid[:,:] = self.EMPTY_CELL

    def place_obstacle(self, coord):
        self.cover_cell(coord, self.OBSTACLE_CELL)

    def off_grid(self, coord): # Where coord is off the grid or not.. 
        return coord[0]<0 or coord[0]>=self.height or coord[1]<0 or coord[1]>=self.width

    def diag_collision(self, prev_pos, new_pos, direction): # TO DO: Test if this works... perhaps better in pcb_envs.py
        # TO DO: Refactor!! This could be written much better lol :) ... 
        if direction==1 or direction==3:
            adj1 = (new_pos[0],new_pos[1]-1)
            adj2 = (prev_pos[0],prev_pos[1]+1)
        elif direction==5 or direction==7: 
            adj1 = (new_pos[0],new_pos[1]+1)
            adj2 = (prev_pos[0],prev_pos[1]-1)
        else: 
            return False 

        if self.cell_val(adj1)==self.cell_val(adj2) and self.cell_val(adj1)!=self.EMPTY_CELL:
            return True
        else :
            return False 

    def cover_cell(self, coord, obj):
        self.grid[coord] = obj 

    def cell_val(self, coord):
        return self.grid[coord]