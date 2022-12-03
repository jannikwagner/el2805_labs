# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
import random
from collections import deque
from networks import Network1
import copy


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


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
# Number of episodes, recommended: 100 - 1000
N_episodes = 400
gamma = 0.95                       # Value of the discount factor
epsilon = 0.1  # exploration param TODO: decreasing epsilon
alpha = 0.001  # learning rate, recommended: 0.001 - 0.0001
batch_size = 32  # batch size N, recommended: 4 − 128
# replay buffer size L, recommended: 5000 - 30000
buffer_size = 10000
# C: Number of episodes between each update of the target network
target_period = int(buffer_size / batch_size)
CLIPPING_VALUE = 1.0  # recommended: 0.5 - 2.0

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
n_ep_running_average = 50                    # Running average of 50 episodes

device = "cuda" if torch.cuda.is_available() else "cpu"

network = Network1(dim_state, n_actions, hidden_size=32).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda k: 1)


def dqn(env: gym.Env,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gamma: float,
        buffer_size: int,
        N_episodes: int,
        target_period: int,
        batch_size: int,
        epsilon: float):

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # initialize the replay buffer
    replay_buffer = ReplayBuffer(buffer_size, device)
    target_network = copy.deepcopy(network).to(device).eval()

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    step = 0
    for k in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        s = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            step += 1
            if step % target_period == 0:
                target_network = copy.deepcopy(network).to(device).eval()
            if random.random() < epsilon:
                a = random.randint(0, n_actions-1)
            else:
                Q_s = network(torch.as_tensor(
                    s, dtype=torch.float32).to(device))
                arg_max = torch.where(Q_s == Q_s.max())[0]
                i = random.randint(0, len(arg_max)-1)
                a = arg_max[i].item()

            # Get next state and reward.
            next_s, r, done, _ = env.step(a)
            obs = (s, a, r, next_s, done)
            replay_buffer.add(obs)

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, done_list = replay_buffer.get_batch(
                    batch_size)
                optimizer.zero_grad()
                Q_theta = network(states)
                Q_phi = target_network(next_states)
                y = rewards + (gamma *
                               torch.max(Q_phi, dim=1)[0] * (1-done_list.int()))
                loss = torch.nn.functional.mse_loss(y,
                                                    Q_theta[range(batch_size), actions.numpy()])

                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm(
                    network.parameters(), CLIPPING_VALUE)

            # Update episode reward
            total_episode_reward += r

            # Update state for next iteration
            s = next_s
            t += 1

        scheduler.step()
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - lr: {}".format(
                k, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps,
                                n_ep_running_average)[-1],
                scheduler.get_lr()[0]))
    return episode_reward_list, episode_number_of_steps


episode_reward_list, episode_number_of_steps = dqn(
    env, network, optimizer, scheduler, gamma, buffer_size, N_episodes, target_period, batch_size, epsilon)

nn_file_name = "neural-network-1.pth"
torch.save(network.to("cpu").state_dict(), nn_file_name)


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)],
           episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)],
           episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
