# %% [markdown]
# # Lab 1 - Task 1 - Bonus
#
# Students: Jannik Wagner (19971213-1433) and Lea Keller (19980209-4889)
#
# Task 1 can be found in [../problem_1.ipynb](../problem_1.ipynb)

# %%
import maze2 as mz2
import maze as mz
import numpy as np
import matplotlib.pyplot as plt
import qlearning as ql
from sarsa import sarsa
from MC import mc
import tqdm
from utility import monte_carlo_success

# %%
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
env = mz2.Maze(maze)
env_p0 = mz2.Maze(maze, poison_prob=0)

# %%
f1 = "results/VI_m2_p0_g0.98.npy"

# %%
# Discount Factor
gamma = 0.98
# Accuracy treshold
epsilon = 0.0001
V, policy_1 = mz2.value_iteration(env_p0, gamma, epsilon)
np.save(f1, (V))

# %%
V = np.load(f1, allow_pickle=True)
policy_1 = env.transition_probabilities.transpose(
    1, 2, 0).dot(V).argmax(axis=1)

# %%
print(monte_carlo_success(policy_1, env))
print(V[env.start_state()], V[env.winning_state_representative])

# %%
f2 = "results/q50k_m2_p0_eps0.2_g0.98.npy"

# %%
alpha = 2/3
epsilon = 0.2
n_episodes = 50000
Q_2, policy_2, v_start_2 = ql.qlearning(
    env_p0, gamma, alpha, epsilon, n_episodes)
np.save(f2, (Q_2))

# %%
Q_2 = np.load(f2, allow_pickle=True)
policy_2 = Q_2.argmax(axis=1)

# %%
print(monte_carlo_success(policy_2, env))
print(Q_2[env.start_state()].max())

# %%
f3 = "results/q50k_m2_p0_eps0.1_g0.98.npy"

# %%
alpha = 2/3
epsilon = 0.1
n_episodes = 50000
Q_3, policy_3, v_start_3 = ql.qlearning(
    env_p0, gamma, alpha, epsilon, n_episodes)
np.save(f3, (Q_3))

# %%
Q_3 = np.load(f3, allow_pickle=True)
policy_3 = Q_3.argmax(axis=1)

# %%
print(monte_carlo_success(policy_3, env))
print(Q_3[env.start_state()].max())

# %%
plt.plot(v_start_2, label="eps=0.2")
plt.plot(v_start_3, label="eps=0.1")
plt.xlabel("Episode")
plt.ylabel("V(s_0)")
plt.hlines(V[env.start_state()], 0, n_episodes, label="VI")
plt.legend()
plt.title("Q-learning, eps=0.2 vs eps=0.1")
plt.savefig("1.i.b.pdf")

# %%
epsilon = 0.2

# %%
f4 = "results/q50k_m2_p0_eps0.2_g0.98_a0.6.npy"

# %%
alpha = 0.6
n_episodes = 50000
Q_4, policy_4, v_start_4 = ql.qlearning(
    env_p0, gamma, alpha, epsilon, n_episodes)
np.save(f4, (Q_4))

# %%
Q_4 = np.load(f4, allow_pickle=True)
policy_4 = Q_4.argmax(axis=1)

# %%
print(monte_carlo_success(policy_4, env))
print(Q_4[env.start_state()].max())

# %%
f5 = "results/q50k_m2_p0_eps0.2_g0.98_a0.9.npy"

# %%
alpha = 0.9
n_episodes = 50000
Q_5, policy_5, v_start_5 = ql.qlearning(
    env_p0, gamma, alpha, epsilon, n_episodes)
np.save(f5, (Q_5))

# %%
Q_5 = np.load(f5, allow_pickle=True)
policy_5 = Q_5.argmax(axis=1)

# %%
print(monte_carlo_success(policy_5, env))
print(Q_5[env.start_state()].max())

# %%
plt.plot(v_start_4, label="alpha=0.6")
plt.plot(v_start_5, label="alpha=0.9")
plt.xlabel("Episode")
plt.ylabel("V(s_0)")
plt.hlines(V[env.start_state()], 0, n_episodes, label="VI")
plt.legend()
plt.title("Q-learning, alpha=0.6 vs alpha=0.9")
plt.savefig("1.i.c.pdf")

# %% [markdown]
# ## 1.j

# %%
f6 = "results/sarsa50k_m2_p0_eps0.2_g0.98.npy"

# %%
gamma = 0.98
alpha = 2/3
epsilon = 0.2
n_episodes = 50000
Q_6, policy_6, v_start_6 = sarsa(
    env_p0, gamma, alpha, epsilon, n_episodes, eps_mode=0)
np.save(f6, (Q_6))

# %%
Q_6 = np.load(f6, allow_pickle=True)
policy_6 = Q_6.argmax(axis=1)

# %%
print(monte_carlo_success(policy_6, env))
print(Q_6[env.start_state()].max())

# %%
f7 = "results/sarsa50k_m2_p0_eps0.1_g0.98.npy"

# %%
alpha = 2/3
epsilon = 0.1
n_episodes = 50000
Q_7, policy_7, v_start_7 = sarsa(
    env_p0, gamma, alpha, epsilon, n_episodes, eps_mode=0)
np.save(f7, (Q_7))

# %%
Q_7 = np.load(f7, allow_pickle=True)
policy_7 = Q_7.argmax(axis=1)

# %%
print(monte_carlo_success(policy_7, env))
print(Q_7[env.start_state()].max())

# %%
plt.plot(v_start_6, label="eps=0.2")
plt.plot(v_start_7, label="eps=0.1")
plt.xlabel("Episode")
plt.ylabel("V(s_0)")
plt.hlines(V[env.start_state()], 0, n_episodes, label="VI")
plt.legend()
plt.title("SARSA, eps=0.2 vs eps=0.1")
plt.savefig("1.j.b.pdf")

# %%
epsilon = 0.2

# %%
f8 = "results/sarsa50k_m2_p0_eps0.2_g0.98_d0.6.npy"

# %%
alpha = 2/3
n_episodes = 50000
delta = 0.6
Q_8, policy_8, v_start_8 = sarsa(
    env_p0, gamma, alpha, epsilon, n_episodes, delta=delta, eps_mode=2)
np.save(f8, (Q_8))

# %%
Q_8 = np.load(f8, allow_pickle=True)
policy_8 = Q_8.argmax(axis=1)

# %%
print(monte_carlo_success(policy_8, env))
print(Q_8[env.start_state()].max())

# %%
f9 = "results/q50k_m2_p0_eps0.2_g0.98_d0.9.npy"

# %%
alpha = 2/3
delta = 0.9
n_episodes = 50000
Q_9, policy_9, v_start_9 = sarsa(
    env_p0, gamma, alpha, epsilon, n_episodes, delta=delta, eps_mode=2)
np.save(f9, (Q_9))

# %%
Q_9 = np.load(f9, allow_pickle=True)
policy_9 = Q_9.argmax(axis=1)

# %%
print(monte_carlo_success(policy_9, env))
print(Q_9[env.start_state()].max())

# %%
plt.plot(v_start_8, label="delta=0.6")
plt.plot(v_start_9, label="delta=0.9")
plt.xlabel("Episode")
plt.ylabel("V(s_0)")
plt.hlines(V[env.start_state()], 0, n_episodes, label="VI")
plt.legend()
plt.title("SARSA, delta=0.6 vs delta=0.9")
plt.savefig("1.j.c.pdf")

# %%
print(monte_carlo_success(policy_4, env, N=10000))
print(Q_4[env.start_state()].max(),
      Q_4[env.winning_state_representative].max())

# %%
print(monte_carlo_success(policy_6, env, N=10000))
print(Q_6[env.start_state()].max(),
      Q_6[env.winning_state_representative].max())

# %%
print(monte_carlo_success(policy_1, env, N=10000))
print(V[env.start_state()], V[env.winning_state_representative])

# %%
V[env.start_state()] / V[env.winning_state_representative]

# %%
Q_6[env.start_state()].max() / Q_6[env.winning_state_representative].max()

# %%
V[env.winning_state_representative]*(1-gamma)

# %%
# env_old = mz.Maze(maze)
env_old_p0 = mz.Maze(maze, poison_prob=0)
f10 = "results/VI_m_p0_g0.98.npy"
# Discount Factor
gamma = 0.98
# Accuracy treshold
epsilon = 0.0001
V_2, policy_V2 = mz.value_iteration(env_p0, gamma, epsilon)
np.save(f10, (V))
V = np.load(f10, allow_pickle=True)
policy_V2 = env.transition_probabilities.transpose(
    1, 2, 0).dot(V).argmax(axis=1)
