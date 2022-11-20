import maze2 as mz
import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql

# Create a maze
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 3],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])
# maze = np.array([
#     [0,  0, 0, 3],
#     [0,  0, 0, 0],
#     [0,  2, 0, 0]
# ])

env = mz.Maze(maze)
# Discount Factor
gamma = 0.95
# Accuracy treshold
epsilon = 0.01
print("value iteration")
V, policy = mz.value_iteration(env, gamma, epsilon)
method = "ValIter"
path = env.simulate((0, 0), policy, method)
print(path, len(path))
gamma = 0.95
alpha = 0.1
epsilon = 1/3
n_episodes = 5000
Q, policy = ql.qlearning(env, gamma, alpha, epsilon, n_episodes)
np.save("test.npz", Q)
method = "ValIter"
path = env.simulate((0, 0), policy, method)
print(path, len(path))
# mz.animate_solution(maze, path)
