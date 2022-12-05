import numpy as np
import torch
import random
from collections import deque
import matplotlib.pyplot as plt


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

# Plot Rewards and steps


def plot_training(n_ep_running_average, episode_reward_list, episode_number_of_steps, plot_path):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    n = len(episode_reward_list)
    x = range(1, n+1)
    ax[0].plot(x, episode_reward_list, label='Episode reward')
    ax[0].plot(x, running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(x, episode_number_of_steps, label='Steps per episode')
    ax[1].plot(x, running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.savefig(plot_path)
    plt.show()


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
            np.stack([exp[0] for exp in batch]), device=self.device)
        actions = torch.as_tensor(
            [exp[1]for exp in batch], device=self.device)
        rewards = torch.as_tensor(
            [exp[2] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([exp[3]for exp in batch]), device=self.device)
        done_list = torch.as_tensor(
            [exp[4] for exp in batch], device=self.device)
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
