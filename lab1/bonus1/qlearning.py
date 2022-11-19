from typing import Tuple
import numpy as np
import maze as mz
import matplotlib.pyplot as plt


def qlearning(env: mz.Maze, gamma: float, alpha: float, epsilon: float, n_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
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
    Q = np.zeros((env.n_states, env.n_actions))
    # Initialize policy
    policy = np.zeros((env.nS, env.nA))
    # For each episode
    for episode in range(n_episodes):
        # Reset environment
        state = env.reset()
        # For each step
        while True:
            # Choose action
            action = np.random.choice(env.nA, p=policy[state])
            # Take action
            next_state, reward, done, _ = env.step(action)
            # Update Q
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            # Update policy
            policy[state] = np.eye(env.nA)[np.argmax(Q[state])]
            # Update state
            state = next_state
            # If done
            if done:
                break
    return Q, policy
