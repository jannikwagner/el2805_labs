import maze as mz
import numpy as np
import matplotlib.pyplot as plt

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

env = mz.Maze(maze)
# Discount Factor
gamma = 0.95
# Accuracy treshold
epsilon = 0.0001
T = 15
V, policy = mz.value_iteration(env, gamma, epsilon)
# method = 'DynProg'
method = "ValIter"
path = env.simulate((0, 0), policy, method)
print(path)
# mz.animate_solution(maze, path)
