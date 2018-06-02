from collections import deque
import numpy as np
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, sample_bias):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.sample_bias = sample_bias
        self.buffer = list()

    def getBatch(self, batch_size):
        if self.num_experiences <= batch_size:
            return self.buffer, self.num_experiences
        else:
            # the mini-batch has to be in time-order, batch is selected according weighted probability
            random_batch = self.weighted_sample(self.num_experiences-batch_size, self.sample_bias)
            # print(self.num_experiences-batch_size, random_batch)
            return self.buffer[random_batch : random_batch + batch_size], batch_size

    def weighted_sample(self, size, sample_bias):
        # the samples are drawn according to weighting of [1,2,3,4......100], latest experiences has high chance to be drawn, vice verse
        weight = np.array([sample_bias**i for i in range(1,size+1)])
        return np.random.choice(size, 1, p=weight / sum(weight))[0]

    def size(self):
        return self.buffer_size

    def add(self, state, future_price, last_action, prediction):
        experience = (state, future_price, last_action, prediction)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = list()
        self.num_experiences = 0

