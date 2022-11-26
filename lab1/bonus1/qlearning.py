from typing import Tuple
import numpy as np
import maze as mz
import matplotlib.pyplot as plt
import tqdm


def qlearning(env: mz.Maze, gamma: float, alpha: float, epsilon: float, n_episodes: int, Q=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Q-Learning algorithm
    :param env: environment
    :param gamma: discount factor
    :param alpha: learning rate
    :param epsilon: exploration rate
    :param n_episodes: number of episodes
    :return: Q, policy, v_start
    """
    # Initialize Q
    if Q is None:
        Q = np.zeros((env.n_states, env.n_actions))
        Q[:, 1:] = 1 - gamma
        Q[:, 0] = -(1 - gamma)
    # count state action appearances
    N = np.zeros((env.n_states, env.n_actions), dtype=int)
    v_start = np.zeros(n_episodes)
    for episode in tqdm.tqdm(range(n_episodes)):
        # Reset environment
        s = env.reset()
        v_start[episode] = Q[s].max()
        while True:
            # Choose action
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            else:
                a = np.random.choice(np.where(Q[s] == Q[s].max())[0])
            N[s, a] += 1
            # Take action
            s_tp1, r, done, _ = env.step(s, a)
            # Update Q
            Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * \
                (r + gamma * np.max(Q[s_tp1]) - Q[s, a])
            # Update state
            s = s_tp1
            # If done
            if done:
                break
    # Initialize policy
    policy = Q.argmax(axis=1)
    return Q, policy, v_start
