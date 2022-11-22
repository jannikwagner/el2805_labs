from typing import Tuple
import numpy as np
import maze as mz
import matplotlib.pyplot as plt
import tqdm


def mc(env: mz.Maze, gamma: float, alpha: float, epsilon: float, n_episodes: int, Q=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MC algorithm
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
        obs = []
        # Reset environment
        s = env.reset()
        # For each step
        q_start[episode] = Q[s].max()
        while True:
            # Choose action
            a = np.random.choice(np.where(Q[s] == Q[s].max())[0])
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            # Take action
            next_s, r, done, _ = env.step(s, a)
            # Update state
            obs.append((s, a, r))
            s = next_s
            # If done
            if done:
                break
        # Update Q
        G = 0
        G = Q[next_s].max()  # STRICTLY NOT MC
        Gs = dict()
        for s, a, r in reversed(obs):
            G = gamma * G + r
            Gs[(s, a)] = G
            # Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * (G - Q[s, a])
        for (s, a) in Gs:
            N[s, a] += 1
            Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * (Gs[(s, a)] - Q[s, a])

    # Initialize policy
    policy = Q.argmax(axis=1)
    return Q, policy, q_start
