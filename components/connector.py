import numpy as np

class Connector():
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3  
    SOUTH = 4 
    SOUTHWEST = 5 
    WEST = 6 
    NORTHWEST = 7 

    def __init__(self, start_coord, idx): 
        self.id = idx 
        self.trail = self.id
        self.head = self.id + 0.3 
        self.goal = self.id + 0.6
        self.connection = [start_coord] 

    def get_head(self): 
        return self.connection[-1]
    
    def move(self, direction):
        self.connection.append(self._step(direction))

    def pop_head(self):
        self.connection.pop()

    def _step(self, direction):
        current_pos = self.get_head()

        if direction == self.NORTH:
            return (current_pos[0]-1, current_pos[1])

        elif direction == self.NORTHEAST:
            return (current_pos[0]-1, current_pos[1]+1)

        elif direction == self.EAST:
            return (current_pos[0], current_pos[1]+1)

        elif direction == self.SOUTHEAST:
            return (current_pos[0]+1, current_pos[1]+1)

        elif direction == self.SOUTH:
            return (current_pos[0]+1, current_pos[1])        

        elif direction == self.SOUTHWEST:
            return (current_pos[0]+1, current_pos[1]-1)

        elif direction == self.WEST:
            return (current_pos[0], current_pos[1]-1)

        elif direction == self.NORTHWEST:
            return (current_pos[0]-1, current_pos[1]-1)  




