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

# if __name__  == '__main__':
#     import matplotlib.pyplot as plt
#     a_1 = np.array([1**i for i in range(1,190+1)]) / sum(np.array([1**i for i in range(1,190+1)]))
#     a_5 = np.array([1.02**i for i in range(1,190+1)]) / sum(np.array([1.02**i for i in range(1,200-10+1)]))
#     a_2 = np.array([1.05**i for i in range(1,190+1)]) / sum(np.array([1.05**i for i in range(1,190+1)]))
#     a_3 = np.array([1.1**i for i in range(1,190+1)]) / sum(np.array([1.1**i for i in range(1,190+1)]))
#     a_4 = np.array([1.15**i for i in range(1,190+1)]) / sum(np.array([1.15**i for i in range(1,190+1)]))
#     t = np.array([i for i in range(1,190+1)])
#
#     plt.title('Probability Distribution (Buffer size = 200, Batch size = 10)')
#     plt.xlabel('i-th mini-batch')
#     plt.plot(t,a_1)
#     plt.plot(t,a_5)
#     plt.plot(t,a_2)
#     plt.plot(t,a_3)
#     # plt.plot(t,a_4)
#     plt.legend(['s = 1','s = 1.02','s = 1.05','s = 1.1','s = 1.15'])
#     plt.show()

