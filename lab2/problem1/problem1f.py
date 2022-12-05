import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import numpy as np
import gym
import torch
import math

env = gym.make('LunarLander-v2')
env.reset()

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

device = "cuda" if torch.cuda.is_available() else "cpu"

exp = "DQN8"
submission_file = f"weights/{exp}.pth"
network = torch.load(submission_file).to(device).eval()

y_min, y_max = 0, 1.5
w_min, w_max = -math.pi, math.pi
res = 50

ys = np.linspace(y_min, y_max, res)
ws = np.linspace(w_min, w_max, res)

Y, W = np.meshgrid(ys, ws)
ZERO = np.zeros_like(Y)
S = np.stack([ZERO, Y, ZERO, ZERO, W, ZERO, ZERO, ZERO],
             axis=2).reshape(res*res, 8)

Q = network(torch.tensor(S, dtype=torch.float32,
            device=device)).cpu().detach().numpy()
Q = Q.reshape(res, res, n_actions)
V = np.max(Q, axis=-1)
A = np.argmax(Q, axis=-1)

ax = plt.axes(projection='3d')
ax.plot_surface(Y, W, V, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title(f'Value function of {exp}')
ax.set_xlabel('y')
ax.set_ylabel('w')
ax.set_zlabel('V(s)')

plt.show()

ax = plt.axes(projection='3d')
# ax.plot_surface(Y, W, A, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
ax.scatter(Y, W, A, c=A, cmap='viridis', linewidth=0.5)

ax.set_title(f'Action landscape of {exp}')
ax.set_xlabel('y')
ax.set_ylabel('w')
ax.set_zlabel('action')

plt.show()