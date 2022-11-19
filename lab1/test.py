import maze as mz
import numpy as np
import matplotlib.pyplot as plt

# Create a maze
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = mz.Maze(maze, poison_prob=1/10)
T = 15
V, policy = mz.dynamic_programming(env, T)
method = 'DynProg'
path = env.simulate((0, 0), policy, method)
print(path)
# mz.animate_solution(maze, path)
