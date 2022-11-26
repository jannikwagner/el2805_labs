from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def sarsa(env, gamma: float, alpha: float, epsilon: float, n_episodes: int, delta=1, eps_mode=0, Q=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SARSA algorithm
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
    N = np.zeros((env.n_states, env.n_actions), dtype=int)
    v_start = np.zeros(n_episodes)

    for episode in tqdm.tqdm(range(n_episodes)):
        if eps_mode == 1:
            epsilon_episode = epsilon * (1-episode/n_episodes)
        elif eps_mode == 2:
            epsilon_episode = 1 / (1+episode)**delta
        else:
            epsilon_episode = epsilon
        # Reset environment
        s = env.reset()
        a = choose_action(env, Q, epsilon_episode, s)
        v_start[episode] = Q[s].max()
        while True:
            # Choose action
            N[s, a] += 1
            # Take action
            s_tp1, r, done, _ = env.step(s, a)
            a_tp1 = choose_action(env, Q, epsilon_episode, s_tp1)
            # Update Q
            Q[s, a] = Q[s, a] + 1/N[s, a]**alpha * \
                (r + gamma * Q[s_tp1, a_tp1] - Q[s, a])
            # Update state
            s = s_tp1
            a = a_tp1
            # If done
            if done:
                break
    # Initialize policy
    policy = Q.argmax(axis=1)
    return Q, policy, v_start


def choose_action(env, Q, epsilon_episode, s):
    if np.random.random() < epsilon_episode:
        a = np.random.randint(env.n_actions)
    else:
        a = np.random.choice(np.where(Q[s] == Q[s].max())[0])
    return a
