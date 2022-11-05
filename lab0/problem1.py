import itertools
import numpy as np
import matplotlib.pyplot as plt

X_MIN = 0
Y_MIN = 0

X_MAX = 6
Y_MAX = 5

WIDTH = X_MAX - X_MIN + 1
HEIGHT = Y_MAX - Y_MIN + 1
NUM_STATES = WIDTH * HEIGHT
NUM_ACTIONS = 5


def dynamic_programming(P, R, T):
    u = R.max(axis=1)
    for t in range(T-1):
        u = (R + P @ u).max(axis=1)
    return u


def value_iteration(P, R, lamda, epsilon=0.01, T=10**10):
    delta = np.inf
    deltas = []
    V = R.max(axis=1)
    values = [V]
    for t in range(T-1):
        V_new = (R + lamda * P @ V).max(axis=1)
        delta = euclidean(V-V_new)

        V = V_new
        values.append(V_new)
        deltas.append(delta)
        if delta <= epsilon*(1-lamda)/lamda:
            break
    return V, values, deltas


def euclidean(x):
    return np.sqrt(np.sum(np.square(x)))


def get_policy(P, R, V):
    pi = (R + P @ V).argmax(axis=1)
    return pi


def apply_policy(P, R, pi, s0, T):
    n_states = P.shape[0]
    s = s0
    states = [s0]
    actions = []
    rewards = []
    r_sum = 0
    for t in range(T):
        a = pi[s]
        r = R[s, a]
        r_sum += r
        s_next = np.random.choice(n_states, p=P[s, a, :])
        s = s_next

        actions.append(a)
        rewards.append(r)
        states.append(s_next)
    return states, actions, r_sum


class MDP:
    def __init__(self, P: np.ndarray, R: np.ndarray):
        assert len(P.shape) == 3
        assert len(R.shape) == 2
        i, j, m = P.shape  # (s, a, s')
        n, l = R.shape  # (s, a)
        assert i == m == n
        assert j == l

        self.P = P
        self.R = R
        self.num_states = n
        self.num_actions = l

    def dynamic_programming(self, T):
        return dynamic_programming(self.P, self.R, T)

    def get_policy(self, V):
        return get_policy(self.P, self.R, V)

    def apply_policy(self, pi, s0, T):
        return apply_policy(self.P, self.R, pi, s0, T)

    def value_iteration(self, lamda, epsilon=0.01, T=10**10):
        return value_iteration(self.P, self.R, lamda, epsilon, T)


def get_probability_a(x, y, a, x_next, y_next, obstacles):
    x_new, y_new = apply_action(x, y, a, obstacles)

    if (x_new, y_new) == (x_next, y_next):
        return 1

    return 0


def outside_map(x, y):
    return x < 0 or y < 0 or x > X_MAX or y > Y_MAX


def apply_action(x, y, a, obstacles):
    x_new, y_new = x, y
    if a == 0:  # up
        y_new -= 1
    elif a == 1:  # right
        x_new += 1
    elif a == 2:  # down
        y_new += 1
    elif a == 3:  # left
        x_new -= 1
    elif a == 4:  # stay
        pass
    else:
        raise RuntimeError

    move_illegal = outside_map(x_new, y_new) or (x_new, y_new) in obstacles
    if move_illegal:
        return x, y

    return x_new, y_new


def state_to_coordinates(s):
    x = s // HEIGHT
    y = s % HEIGHT
    return (x, y)


if __name__ == "__main__":

    # a) Define The MDP
    obstacles = {(2, 0), (2, 1), (2, 2), (1, 4),
                 (2, 4), (3, 4), (4, 4), (5, 4)}

    P_: np.ndarray = np.zeros((WIDTH, HEIGHT, NUM_ACTIONS, WIDTH, HEIGHT))
    for x, y, a, x_next, y_next in itertools.product(range(WIDTH), range(HEIGHT), range(NUM_ACTIONS), range(WIDTH), range(HEIGHT)):
        P_[x, y, a, x_next, y_next] = get_probability_a(
            x, y, a, x_next, y_next, obstacles)

    P = P_.reshape(NUM_STATES, NUM_ACTIONS, WIDTH, HEIGHT).reshape(
        NUM_STATES, NUM_ACTIONS, NUM_STATES)

    NEGATIVE_REWARD = - 0.01
    POSITIVE_REWARD = 1

    R_ = np.zeros((WIDTH, HEIGHT, NUM_ACTIONS))

    A = (0, 0)
    B = (5, 5)
    R_ = R_ + NEGATIVE_REWARD

    for x, y, a in itertools.product(range(WIDTH), range(HEIGHT), range(NUM_ACTIONS)):
        x_new, y_new = apply_action(x, y, a, obstacles)
        if (x_new, y_new) == B:
            R_[x, y, a] = POSITIVE_REWARD

        # with more time we don't want to keep collecting reward
        # thus the reward of staying on B should be 0
        # this means that we have to punish moving away from B
        # (with more than the reward for moving there)
        # so that we do not alternate between B and its neighbours
        if (x, y) == B:
            R_[x, y, a] = - 2*POSITIVE_REWARD  # punish moving away
            if a == 4:
                R_[x, y, a] = 0  # no reward for staying

    R = R_.reshape(NUM_STATES, NUM_ACTIONS)

    # print(R_[:, :, 0].T)
    # print(R_[:, :, 1].T)
    # print(R_[:, :, 2].T)
    # print(R_[:, :, 3].T)
    # print(R_[:, :, 4].T)

    mdp = MDP(P, R)

    # b) solve using dynamic programming
    print("b) solve using dynamic programming")
    T = WIDTH * HEIGHT + 1  # in our case, 5+5+1 is sufficient
    V = mdp.dynamic_programming(T)
    print("V:\n", V.reshape(WIDTH, HEIGHT).T)

    pi = mdp.get_policy(V)
    print("pi:\n", pi.reshape(WIDTH, HEIGHT).T)

    states, actions, r_sum = mdp.apply_policy(pi, 0, T)
    print("action sequence:\n", actions)
    print("state sequence:\n", [state_to_coordinates(s) for s in states])

    # c) Value iteration
    print()
    print("c) Value iteration")
    lamda = 0.99
    epsilon = 0.01
    V, values, deltas = mdp.value_iteration(lamda, epsilon)
    print("V:\n", V.reshape(WIDTH, HEIGHT).T)
    print("deltas:\n", deltas)
    values = np.array(values)
    T, S = values.shape

    for s in range(NUM_STATES):
        plt.plot(range(T), values[:, s], label=state_to_coordinates(s))
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("V(s)")
    plt.show()

    # d) modified problem
