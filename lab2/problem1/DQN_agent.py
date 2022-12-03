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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import copy
import torch
from utils import ReplayBuffer, EpsilonDecay, running_average
import random

CLIPPING_VALUE = 1.0  # recommended: 0.5 - 2.0


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self, *args):
        ''' Performs a backward pass on the network '''
        pass

    def episode_start(self):
        ''' Called at the beginning of each episode '''
        pass

    def status_text(self):
        ''' Returns a string to be printed on the console '''
        return ''


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class DQNAgent(Agent):
    def __init__(self,
                 n_actions: int,
                 network: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 replay_buffer: ReplayBuffer,
                 epsilon_decay: EpsilonDecay,
                 device: str,
                 gamma=0.99, batch_size=32, target_period=1000):
        super(DQNAgent, self).__init__(n_actions)
        self.network = network.to(device)
        self.target_network = copy.deepcopy(network)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = torch.nn.MSELoss()
        self.target_period = target_period
        self.step = 0
        self.k = 0
        self.epsilon_decay = epsilon_decay

    def episode_start(self):
        ''' Called at the beginning of each episode '''
        self.epsilon = self.epsilon_decay.get(self.k)
        if self.k != 0:
            self.scheduler.step()
        self.k += 1
        super().episode_start()

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action using the epsilon-greedy policy

            Returns:
                action (int): the action
        '''
        if random.random() < self.epsilon:
            a = random.randint(0, self.n_actions-1)
        else:
            Q_s = self.network(torch.as_tensor(
                state, dtype=torch.float32).to(self.device))
            arg_max = torch.where(Q_s == Q_s.max())[0]
            i = random.randint(0, len(arg_max)-1)
            a = arg_max[i].item()

        self.last_state = state
        self.last_action = a
        return a

    def backward(self, next_s: np.ndarray, r: float, done: bool):
        ''' Performs a backward pass on the network '''
        self.step += 1
        if self.step % self.target_period == 0:
            self.target_network = copy.deepcopy(
                self.network).to(self.device).eval()

        obs = (self.last_state, self.last_action, r, next_s, done)
        self.replay_buffer.add(obs)

        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, done_list = self.replay_buffer.get_batch(
                self.batch_size)
            self.optimizer.zero_grad()
            Q_theta = self.network(states)
            Q_phi = self.target_network(next_states)
            y = rewards + (self.gamma *
                           torch.max(Q_phi, dim=1)[0] * (1-done_list.int()))
            loss = self.loss(y,
                             Q_theta[range(Q_theta.size()[0]), actions.numpy()])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), CLIPPING_VALUE)
            self.optimizer.step()

    def status_text(self):
        ''' Returns a string to be printed on the console '''
        return "lr: {} - eps: {}".format(
            round(self.scheduler.get_last_lr()[0], 5),
            round(self.epsilon, 3))
