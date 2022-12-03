import gym
from DQN_agent import Agent
from tqdm import trange
from utils import running_average
import numpy as np


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
