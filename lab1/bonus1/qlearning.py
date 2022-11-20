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
        Q[:, 1:] = 1
        Q[:, 0] = -1
    N = np.zeros((env.n_states, env.n_actions), dtype=int)
    # Initialize policy
    policy = Q.argmax(axis=1)
    # For each episode
    for episode in tqdm.tqdm(range(n_episodes)):
        # print("episode =", episode)
        # Reset environment
        s = env.reset()
        # For each step
        while True:
            # Choose action
            a = policy[s]
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            N[s, a] += 1
            # Take action
            next_s, r, done, _ = env.step(s, a)
            # Update Q
            Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * \
                (r + gamma * np.max(Q[next_s]) - Q[s, a])
            # Update policy
            policy[s] = np.argmax(Q[s])
            # Update state
            # if env.win(next_s):
            #     print(env.states[s], a, r, env.states[next_s])
            s = next_s
            # If done
            if done:
                break
    return Q, policy
