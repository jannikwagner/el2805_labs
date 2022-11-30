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


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 100                             # Number of episodes
gamma = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
# Number of episodes between each update of the target network
target_period = 100
batch_size = 32  # batch size
epsilon = 0.1  # exploration param
buffer_size = 10000                          # replay buffer size
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, n):
        indices = np.random.choice(len(self), n, replace=False)
        return [self.buffer[i] for i in indices]

    def get_batch(self, batch_size):
        batch = self.sample(batch_size)
        states = torch.as_tensor(np.array([exp[0] for exp in batch]))
        actions = torch.as_tensor(np.array([exp[1] for exp in batch]))
        rewards = torch.as_tensor(
            np.array([exp[2] for exp in batch], dtype=np.float32))
        next_states = torch.as_tensor(np.array([exp[3] for exp in batch]))
        done_list = torch.as_tensor(np.array([exp[4] for exp in batch]))
        return states, actions, rewards, next_states, done_list


network = Network1(dim_state, n_actions, hidden_size=8)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)


def dqn(env, gamma, buffer_size, N_episodes, target_period, batch_size, epsilon):

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # Random agent initialization
    agent = RandomAgent(n_actions)
    # TODO: initialize theta and phi

    # initialize the replay buffer TODO: gym function?
    # buffer = queue.Queue(maxsize=buffer_size)
    replay_buffer = ReplayBuffer(buffer_size)
    target_network = copy.deepcopy(network)

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for k in EPISODES:
        if k % target_period == 0:
            target_network = copy.deepcopy(network)
        # Reset enviroment data and initialize variables
        done = False
        s = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # TODO: Take epsilon greedy action wrt Q_theta. Maybe gym?
            if random.random() < epsilon:
                a = random.randint(0, n_actions-1)
            else:
                Q_s = network(torch.as_tensor(s, dtype=torch.float32))
                arg_max = torch.where(Q_s == Q_s.max())[0]
                i = random.randint(0, len(arg_max)-1)
                a = arg_max[i].item()
            # a = agent.forward(s)

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
                y = rewards + gamma * \
                    torch.max(Q_phi, dim=1)[0] * (1-done_list.int())
                loss = torch.nn.functional.mse_loss(
                    Q_theta[range(batch_size), actions.numpy()], y)

                loss.backward()
                optimizer.step()

        # Update episode reward
            total_episode_reward += r

        # Update state for next iteration
            s = next_s
            t += 1

    # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

    # Close environment
        env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                k, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    return agent, episode_reward_list, episode_number_of_steps


agent, episode_reward_list, episode_number_of_steps = dqn(
    env, gamma, buffer_size, N_episodes, target_period, batch_size, epsilon)


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
