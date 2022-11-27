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

env = mz.Maze(maze)

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy = mz.dynamic_programming(env, horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start = (0, 0)
path = env.simulate(start, policy, method)

# Show the shortest path
mz.animate_solution(maze, path, "results/1.c")
