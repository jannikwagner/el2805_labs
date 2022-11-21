from typing import Tuple
import numpy as np
import maze as mz
import matplotlib.pyplot as plt
import tqdm


def qlearning(env: mz.Maze, gamma: float, alpha: float, epsilon: float, n_episodes: int, Q=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q-Learning algorithm
    :param env: environment
    :param gamma: discount factor
    :param alpha: learning rate
    :param epsilon: exploration rate
    :param n_episodes: number of episodes
    :return: Q, policy
    """
    # Initialize Q
    if Q is None:
        Q = np.zeros((env.n_states, env.n_actions))
        Q[:, 1:] = 1 - gamma
        Q[:, 0] = -(1 - gamma)
    N = np.zeros((env.n_states, env.n_actions), dtype=int)
    # For each episode
    q_start = np.zeros(n_episodes)
    for episode in tqdm.tqdm(range(n_episodes)):
        # print("episode =", episode)
        # Reset environment
        s = env.reset()
        # For each step
        q_start[episode] = Q[s].max()
        while True:
            # Choose action
            a = np.random.choice(np.where(Q[s] == Q[s].max())[0])
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            N[s, a] += 1
            # Take action
            next_s, r, done, _ = env.step(s, a)
            # Update Q
            Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * \
                (r + gamma * np.max(Q[next_s]) - Q[s, a])
            # Update state
            s = next_s
            # If done
            if done:
                break
    # Initialize policy
    policy = Q.argmax(axis=1)
    return Q, policy, q_start
