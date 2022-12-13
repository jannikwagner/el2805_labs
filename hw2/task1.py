import numpy as np

obs = [
    [1, 2, 600],
    [0, 0, 80],
    [1, 0, 100],
    [0, 1, 60],
    [1, 2, 70],
    [2, 1, 40],
    [0, 0, 20],
    [2, 2, 0]
]


def qlearning(obs, alpha, gamma):
    Q = np.zeros((3, 3))
    for i in range(len(obs)-1):
        s, a, r = obs[i]
        s_next, _, _ = obs[i+1]
        Q[s, a] += alpha * (r + gamma*Q[s_next].max() - Q[s, a])
        print(s, a, r)
        print(i+1)
        print(Q)
    return Q


def sarsa(obs, alpha, gamma):
    Q = np.zeros((3, 3))
    for i in range(len(obs)-1):
        s, a, r = obs[i]
        s_next, a_next, _ = obs[i+1]
        Q[s, a] += alpha * (r + gamma*Q[s_next, a_next] - Q[s, a])
        print(s, a, r)
        print(i+1)
        print(Q)
    return Q


alpha = 0.1
gamma = 0.5
print("Q-learning")
Q = qlearning(obs, alpha, gamma)
print(Q.argmax(axis=1))

print("SARSA")
sarsa(obs, alpha, gamma)
print(Q.argmax(axis=1))
