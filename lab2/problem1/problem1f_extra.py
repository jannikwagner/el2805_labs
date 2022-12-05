import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import numpy as np
import gym
import torch
import math
from DQN_check_solution import simulate
from DQN_agent import SimulationAgent

env = gym.make('LunarLander-v2')
env.reset()

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

device = "cuda" if torch.cuda.is_available() else "cpu"

exp = "DQN19"
submission_file = f"weights/{exp}.pth"
network = torch.load(submission_file).to(device).eval()
total_episode_reward, states, actions, rewards = simulate(
    SimulationAgent(network), env)
states = np.stack(states)

ax = plt.axes(projection='3d')

zline = rewards
xline = states[:, 1]
yline = states[:, 4]

ax.plot3D(xline, yline, zline)

plt.show()

ax = plt.axes(projection='3d')

zline = range(len(rewards))
xline = states[:, 1]
yline = states[:, 4]

ax.plot3D(xline, yline, zline)
plt.show()

# fig, ax1 = plt.subplots()
# ax1.plot(values, label="value")
# ax2 = ax1.twinx()
# ax2.plot(actions, "o", label="action")
# ax3 = ax1.twinx()
# ax3.plot(states[:, 1], label="y")
# ax4 = ax1.twinx()
# ax4.plot(states[:, 5], label="w")

# plt.legend()

# plt.show()
