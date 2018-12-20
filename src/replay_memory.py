import random
import collections as memory
import numpy as np

class Memory:

    def __init__(self, MemorySize, batch_size, act_dim, obs_dim):
        self.memory_size = MemorySize
        self.batch_size = batch_size
        self.container = memory.deque()
        self.container_size = 0
        self.priority = 1
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def get_size(self):
        return self.batch_size

    def size(self):
        return self.container_size

    def select_batch(self, batchSize):
        # random.sample(COLLECTIONS, #ofSAMPLES)
        return random.sample(self.container, batchSize)

    def add(self, experience):

        experience.append(self.priority)
        if self.container_size < self.memory_size:
            self.container.append(experience)
            self.container_size += 1

        else: 
            self.container.popleft()
            self.container.append(experience)

    def transform_sample(self, sample, batch_size):
        
        obs_dim = self.obs_dim
        act_dim = self.act_dim
    
        current_state = [x[0] for x in sample]
        actions =       np.asarray([x[1] for x in sample])
        rewards =       [x[2] for x in sample]
        next_state =    [x[3] for x in sample]
        done =          [x[4] for x in sample]

        current_state = np.resize(current_state, [batch_size, obs_dim])
        actions       = np.resize(actions, [batch_size, act_dim])
        rewards       = np.resize(rewards, [batch_size, act_dim])
        next_state    = np.resize(next_state, [batch_size, obs_dim])
        done          = np.resize(done, [batch_size, act_dim])

        return [current_state, actions, rewards, next_state, done]

    def select_sample(self, batch_size):
        """ Pick samples from container
        """
        #print ("container_size : ", self.container_size)
        sample = random.sample(self.container, batch_size)
        return self.transform_sample(sample, batch_size)

    def clear_memory(self):
        self.container = memory.deque()
        self.container_size = 0
        self.num_experiences = 0
