import numpy as np
import torch
import random
from collections import deque


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


class ReplayBuffer:
    def __init__(self, buffer_size, device, cer=True):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.cer = cer

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, n):
        samples = random.sample(self.buffer, n)
        if self.cer:
            samples[0] = self.buffer[-1]
        return samples

    def get_batch(self, batch_size):
        batch = self.sample(batch_size)
        states = torch.as_tensor(
            np.array([exp[0] for exp in batch]), device=self.device)
        actions = torch.as_tensor(
            np.array([exp[1] for exp in batch]), device=self.device)
        rewards = torch.as_tensor(
            np.array([exp[2] for exp in batch], dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(
            np.array([exp[3] for exp in batch]), device=self.device)
        done_list = torch.as_tensor(
            np.array([exp[4] for exp in batch]), device=self.device)
        return states, actions, rewards, next_states, done_list


class EpsilonDecay:
    def __init__(self, start, end, Z, mode='linear'):
        self.start = start
        self.end = end
        self.Z = Z
        self.mode = mode

    def linear(self, k):
        return max(self.end, self.start - k * (self.start - self.end) / self.Z)

    def exponential(self, k):
        return max(self.end, self.start * (self.end / self.start) ** (k / self.Z))

    def constant(self, k):
        return self.start

    def get(self, k):
        if self.mode == 'linear':
            return self.linear(k)
        elif self.mode == 'exponential':
            return self.exponential(k)
        elif self.mode == 'constant':
            return self.constant(k)
        else:
            raise ValueError('Unknown mode')
