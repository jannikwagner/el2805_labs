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
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
import numpy as np
import gym
from collections import deque
import torch
import random

buffer_size = 10**6
buffer = deque(maxlen=buffer_size)


def sample_from_buffer(buffer, n):
    indices = np.random.choice(len(buffer), n, replace=False)
    return [buffer[i] for i in indices]


class Network(torch.nn.Module):
    def __init__(self, n, m):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(n, 8)
        self.fc2 = torch.nn.Linear(8, m)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


### CREATE RL ENVIRONMENT ###
# Create a CartPole environment
env = gym.make('CartPole-v0')
n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions

nn = Network(n, m)

optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)

num_samples = 3

### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in range(5):
    state = env.reset()                  # Reset environment, returns initial
    # state
    done = False                         # Boolean variable used to indicate if
    # an episode terminated

    while not done:
        env.render()                     # Render the environment
        state_tensor = torch.tensor([state], requires_grad=False)
        action = nn(state_tensor).argmax(1).item()

        if len(buffer) > num_samples:
            optimizer.zero_grad()
            samples = sample_from_buffer(buffer, num_samples)
            state_tensor = torch.tensor([s[0] for s in samples])
            actions = torch.tensor([s[1] for s in samples])
            action_tensor = nn(state_tensor)
            y = action_tensor[actions]
            z = torch.zeros_like(y)
            loss = torch.nn.functional.mse_loss(y, z)

            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1)

        next_state, reward, done, *_ = env.step(action)

        entry = (state, action, reward, next_state, done)
        buffer.append(entry)

        state = next_state

# Close all the windows
env.close()
