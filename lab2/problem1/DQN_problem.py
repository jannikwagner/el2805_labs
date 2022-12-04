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
from DQN_agent import RandomAgent, DQNAgent
from networks import Network1
from utils import ReplayBuffer, EpsilonDecay, plot
from rl import rl
from DQN_check_solution import check_solution


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

submission_file = "neural-network-1.pth"

# Parameters
# Number of episodes, recommended: 100 - 1000
experiment_name = "DQN25"

N_episodes = 200
gamma = 0.99  # Value of the discount factor
epsilon_max = 0.99  # example: 0.99
epsilon_min = 0.3  # example: 0.05
decay_episode_portion = 0.9  # recommended: 0.9 - 0.95
decay_mode = 'linear'  # possible values: 'linear', 'exponential', 'constant'
alpha = 0.0002  # learning rate, recommended: 0.001 - 0.0001

batch_size = 8  # batch size N, recommended: 4 − 128, 64 seems to work well
# replay buffer size L, recommended: 5000 - 30000
buffer_size = 10000
# C: Number of updates between each update of the target network, recommended: L/N
target_period = int(buffer_size / batch_size)
hidden_size = 128
hidden_layers = 2

n_ep_running_average = 50  # Running average of 50 episodes

device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = False

epsilon_decay = EpsilonDecay(
    epsilon_max, epsilon_min, int(decay_episode_portion * N_episodes), mode=decay_mode)

network = Network1(dim_state, n_actions, hidden_size=hidden_size,
                   hidden_layers=hidden_layers).to(device)
network = torch.load(submission_file).to(device) if load_model else network
print(network)

optimizer = torch.optim.Adam(network.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda k: 1)

replay_buffer = ReplayBuffer(buffer_size, device)

agent = DQNAgent(n_actions, network, optimizer, scheduler, replay_buffer,
                 epsilon_decay, device, gamma, batch_size, target_period)

random_agent = RandomAgent(n_actions)


episode_reward_list, episode_number_of_steps = rl(
    env, agent, N_episodes, n_ep_running_average)


plot_folder = "./plots/"
weights_folder = "./weights/"
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)
nn_path = os.path.join(weights_folder, experiment_name + ".pth")
plot_path = os.path.join(plot_folder, experiment_name + ".png")

torch.save(network.to("cpu"), nn_path)

check_solution(network, env, render=True, N_EPISODES=1)
passed = check_solution(network, env, render=False)

if passed:
    torch.save(network.to("cpu"), submission_file)

plot(n_ep_running_average, episode_reward_list,
     episode_number_of_steps, plot_path)
