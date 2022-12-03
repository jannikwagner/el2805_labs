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
import os
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, Agent, DQNAgent
import random
from networks import Network1
import copy
from utils import running_average, ReplayBuffer, EpsilonDecay, plot


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# Parameters
# Number of episodes, recommended: 100 - 1000
N_episodes = 400
gamma = 0.95  # Value of the discount factor
epsilon_max = 0.99
epsilon_min = 0.05
decay_episode_portion = 0.9  # recommended: 0.9 - 0.95
decay_mode = 'exponential'  # possible values: 'linear', 'exponential', 'constant'
epsilon_decay = EpsilonDecay(
    epsilon_max, epsilon_min, int(decay_episode_portion * N_episodes), mode=decay_mode)
alpha = 0.001  # learning rate, recommended: 0.001 - 0.0001

batch_size = 32  # batch size N, recommended: 4 − 128
# replay buffer size L, recommended: 5000 - 30000
buffer_size = 30000
# C: Number of episodes between each update of the target network
target_period = int(buffer_size / batch_size)
n_ep_running_average = 50                    # Running average of 50 episodes

device = "cuda" if torch.cuda.is_available() else "cpu"

network = Network1(dim_state, n_actions, hidden_size=32).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda k: 1)

replay_buffer = ReplayBuffer(buffer_size, device)

agent = DQNAgent(n_actions, network, optimizer, scheduler, replay_buffer,
                 epsilon_decay, device, gamma, batch_size, target_period)

random_agent = RandomAgent(n_actions)


def rl(env: gym.Env,
        agent: Agent,
        N_episodes: int,
        n_ep_running_average: int,):
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for k in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        s = env.reset()
        total_episode_reward = 0.
        t = 0
        agent.episode_start()
        while not done:
            a = agent.forward(s)

            # Get next state and reward.
            next_s, r, done, _ = env.step(a)
            agent.backward(next_s, r, done)

            # Update episode reward
            total_episode_reward += r

            # Update state for next iteration
            s = next_s
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        MIN_EPISODES = 100
        n_success = 50
        avg_success = 50
        if k > MIN_EPISODES:
            if np.mean(episode_reward_list[-n_success:]) > avg_success:
                print("Early stopping SUCCESS")
                break

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - {}".format(
                k, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps,
                                n_ep_running_average)[-1],
                agent.status_text()))
    return episode_reward_list, episode_number_of_steps


episode_reward_list, episode_number_of_steps = rl(
    env, agent, N_episodes, n_ep_running_average)

experiment_name = "DQN1"
plot_folder = "./plots/"
weights_folder = "./weights/"
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)
nn_path = os.path.join(weights_folder, experiment_name + ".pth")
plot_path = os.path.join(plot_folder, experiment_name + ".png")

torch.save(network.to("cpu").state_dict(), nn_path)

plot(n_ep_running_average, episode_reward_list,
     episode_number_of_steps, plot_path)
