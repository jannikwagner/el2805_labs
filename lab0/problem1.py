import itertools
import numpy as np

X_MIN = 0
Y_MIN = 0

X_MAX = 6
Y_MAX = 5

WIDTH = X_MAX - X_MIN + 1
HEIGHT = Y_MAX - Y_MIN + 1
NUM_STATES = WIDTH * HEIGHT
NUM_ACTIONS = 5


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
        u = self.R.max(axis=1)
        for t in range(T-1):
            u = (self.R + self.P @ u).max(axis=1)
        return u


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


if __name__ == "__main__":
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

    R = R_.reshape(NUM_STATES, NUM_ACTIONS)

    print(R_[:, :, 0].T)
    print(R_[:, :, 1].T)
    print(R_[:, :, 2].T)
    print(R_[:, :, 3].T)
    print(R_[:, :, 4].T)

    mdp = MDP(P, R)

    T = WIDTH + HEIGHT + 1
    u = mdp.dynamic_programming(T)
    print(u.reshape(7, 6).T)
