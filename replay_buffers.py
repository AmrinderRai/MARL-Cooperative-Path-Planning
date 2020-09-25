from abc import ABC, abstractmethod 
from collections import deque
import numpy as np
from numpy import copy 
import random

class ReplayBuffer(ABC):
    @abstractmethod
    def add_to_buffer(self):
        pass

    @abstractmethod
    def sample_minibatch(self):
        pass 

class VanillaReplayBuffer(ReplayBuffer):
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add_to_buffer(self, experience):
        self.buffer.append(experience)
        
    def sample_minibatch(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size)
        samples = np.array(self.buffer)[sample_indices]
        return map(list, zip(*samples))

class PrioritizedReplayBuffer(ReplayBuffer):
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
    
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def sample_minibatch(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset 

class DRQN_ReplayBuffer():
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
        return samples

class DRQN_PrioritizedReplayBuffer():
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
    
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def sample_minibatch(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset 

class SharedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)  

    def add_to_buffer(self, experience, idx):
        self.buffer.append((experience, idx))
        
    def sample_minibatch(self, batch_size, idx):
        sample_size = min(len(self.buffer), batch_size)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size)
        samples = np.array(self.buffer)[sample_indices]
        samples_pov = self.switch_perspective(samples, idx)
        return map(list, zip(*samples_pov))
    
    def switch_perspective(self, samples, pov_id):
        pov_id +=1 
        pov_id *=10 

        switched_samples = []

        for sample in samples:
            idx = sample[1]
            idx +=1
            idx *=10 

            pov_state = sample[0][0][0] 
            pov_next_state = sample[0][1][0]

            mapping = {
                idx: pov_id, 
                idx+3: pov_id+3, 
                idx+6: pov_id+6, 
            }

            pov_state_switched = copy(pov_state)
            pov_next_switched = copy(pov_next_state)
            for k,v in mapping.items(): 
                pov_state_switched[pov_state==k] = v 
                pov_state_switched[pov_state==v] = k 
                pov_next_switched[pov_next_state==k] = v 
                pov_next_switched[pov_next_state==v] = k 

            pov_state_switched = (pov_state_switched, sample[0][0][1])
            pov_next_switched = (pov_next_switched, sample[0][1][1])
            
            switched_samples.append((pov_state_switched, pov_next_switched, *sample[0][2:]))

        return switched_samples

class DRQN_SharedReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)  

    def add_to_buffer(self, experiences, idx):
        self.buffer.append((experiences, idx))
        
    def sample_minibatch(self, batch_size, idx):
        sample_size = min(len(self.buffer), batch_size)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size)
        samples = np.array(self.buffer)[sample_indices]
        samples_pov = self.switch_perspective(samples, idx)
        return samples_pov
    
    def switch_perspective(self, episodes, pov_id):
        pov_id +=1 
        pov_id *=10 
        episode_indices = episodes[:,1]
        episodes = episodes[:,0]
    

        switched_episodes = []

        for i, episode in enumerate(episodes):
            switched_samples = []
            idx = episode_indices[i]
            idx +=1
            idx *=10 
            for sample in episode:
                pov_state = sample[0][0] 
                pov_next_state = sample[1][0]

                mapping = {
                    idx: pov_id, 
                    idx+3: pov_id+3, 
                    idx+6: pov_id+6, 
                }

                pov_state_switched = copy(pov_state)
                pov_next_switched = copy(pov_next_state)
                for k,v in mapping.items(): 
                    pov_state_switched[pov_state==k] = v 
                    pov_state_switched[pov_state==v] = k 
                    pov_next_switched[pov_next_state==k] = v 
                    pov_next_switched[pov_next_state==v] = k 
                pov_state_switched = (pov_state_switched, sample[0][1])
                pov_next_switched = (pov_next_switched, sample[1][1])
                
                switched_samples.append((pov_state_switched, pov_next_switched, *sample[2:]))
            switched_episodes.append(switched_samples)
        return switched_episodes

        