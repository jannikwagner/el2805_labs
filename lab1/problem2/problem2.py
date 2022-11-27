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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import itertools
from tqdm import trange
import pickle

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
d = env.action_space.n
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 1000        # Number of episodes to run for training
gamma = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


# my code
eta = np.array([[0, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [0, 2], ])

p = 2
eta = np.array(list(itertools.product(range(p+1), range(p+1))))
m = eta.shape[0]


def get_phi_s(eta, s):
    return np.cos(np.pi * eta @ s)


def get_Q_s(w, phi):
    return w @ phi


def get_Q_sa(w, phi, a):
    return w[a] @ phi


lamda = 0.99
epsilon = 0.2
alpha = 0.001
# Training process


def chhose_action(k, epsilon, Q_s):
    if epsilon > 0 and np.random.random() < epsilon:
        a = np.random.randint(0, k)
    else:
        a = np.random.choice(np.where(Q_s == Q_s.max())[0])
    return a


def sarsa_lambda(env, N_episodes, gamma, eta, lamda, epsilon, alpha, momentum=0.9, nesterov=True):
    clip_threshold = 1
    scale = (eta**2).sum(axis=1)**0.5
    scale[scale == 0] = 1
    k = env.action_space.n      # tells you the number of actions
    m, n = eta.shape
    w = np.random.randn(k, m) * 0.01
    v = np.zeros_like(w)
    for i in trange(N_episodes):
        # Reset enviroment data
        done = False
        s = scale_state_variables(env.reset())
        total_episode_reward = 0.
        z = np.zeros_like(w)
        epsilon_episode = epsilon

        while not done:
            alpha_t = alpha / scale[None, :]
            # Take a random action
            phi_s = get_phi_s(eta, s)
            Q_s = get_Q_s(w, phi_s)
            a = chhose_action(k, epsilon_episode, Q_s)
            Q_sa = Q_s[a]

            z *= gamma * lamda
            z[a] += phi_s * Q_sa
            z = np.clip(z, -clip_threshold, clip_threshold)
            # print("z =\n", z)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            s_tp1, r, done, _ = env.step(a)
            if r != -1:
                print(r)
            s_tp1 = scale_state_variables(s_tp1)

            phi_s_tp1 = get_phi_s(eta, s_tp1)
            Q_s_tp1 = get_Q_s(w, phi_s_tp1)
            a_tp1 = chhose_action(k, epsilon_episode, Q_s_tp1)
            Q_sa_tp1 = Q_s_tp1[a_tp1]

            delta = r + gamma * Q_sa_tp1 - Q_sa
            if momentum != 0:
                v = momentum * v + alpha_t * delta * z
                if nesterov:
                    w += momentum * v + alpha_t * delta * z
                else:
                    w += v
            else:
                w += alpha_t * delta * z
            # print("w =\n", w)
            # Update episode reward
            total_episode_reward += r

            # Update state for next iteration
            s = s_tp1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()


w = sarsa_lambda(env, N_episodes, gamma,
                 eta, lamda, epsilon, alpha, momentum=0.9)

path = 'weights.pkl'
with open(path, "wb") as f:
    pickle.dump(w, f)

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)],
         episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)],
         running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
