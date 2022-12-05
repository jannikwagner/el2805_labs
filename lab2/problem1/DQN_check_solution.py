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
from tqdm import trange
from DQN_agent import SimulationAgent, RandomAgent
from utils import running_average


def load_model(file_path):
    try:
        model = torch.load(file_path)
        print('Network model: {}'.format(model))
    except:
        print(f'File {file_path} not found!')
        exit(-1)
    return model


N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50


def check_solution(agent, env, render=False, N_EPISODES=N_EPISODES, CONFIDENCE_PASS=CONFIDENCE_PASS):
    episode_reward_list = []  # Used to store episodes reward

    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        total_episode_reward, *_ = simulate(agent, env, render)

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        if render:
            print(f'Episode {i} reward: {total_episode_reward}')

    # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
        avg_reward,
        confidence))
    passed = avg_reward - confidence >= CONFIDENCE_PASS
    if passed:
        print('Your policy passed the test!')
    else:
        print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))
    return passed


def simulate(agent, env, render=False):
    done = False
    state = env.reset()
    total_episode_reward = 0.
    states = []
    actions = []
    rewards = []
    if render:
        env.render()
    while True:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        action = agent.forward(torch.as_tensor([state]))

        states.append(state)
        actions.append(action)
        if render:
            env.render()

        if done:
            break

        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward
        rewards.append(reward)

        # Update state for next iteration
        state = next_state
    return total_episode_reward, states, actions, rewards + [0]


if __name__ == "__main__":
    file_path = 'neural-network-1.pth'
    # file_path = 'weights/DQN8.pth'

    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLander-v2')
    env.reset()
    n_actions = env.action_space.n

    # Load model
    model = load_model(file_path)
    agent = SimulationAgent(model)
    random_agent = RandomAgent(n_actions)

    # Parameters
    render = False
    # Reward

    print("DQN8 agent")
    passed = check_solution(agent, env, render, N_EPISODES, CONFIDENCE_PASS)
    print("Random agent")
    passed = check_solution(random_agent, env, render,
                            N_EPISODES, CONFIDENCE_PASS)
