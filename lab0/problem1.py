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
    x_new, y_new = apply_action(x, y, a)

    move_illegal = outside_map(x_new, y_new) or (x_new, y_new) in obstacles
    if move_illegal:
        stayed_at_position = (x, y) == (x_next, y_next)
        if stayed_at_position:
            return 1
        return 0

    if (x_new, y_new) == (x_next, y_next):
        return 1

    return 0


def outside_map(x, y):
    return x < 0 or y < 0 or x > X_MAX or y > Y_MAX


def apply_action(x, y, a):
    if a == 0:  # up
        y -= 1
    elif a == 1:  # right
        x += 1
    elif a == 2:  # down
        y += 1
    elif a == 3:  # left
        x -= 1
    elif a == 4:  # stay
        pass
    else:
        raise RuntimeError
    return x, y


if __name__ == "__main__":
    obstacles = {(2, 0), (2, 1), (2, 2), (4, 1),
                 (4, 2), (4, 3), (4, 4), (4, 5)}

    P_: np.ndarray = np.zeros((WIDTH, HEIGHT, NUM_ACTIONS, WIDTH, HEIGHT))
    for x, y, a, x_next, y_next in itertools.product(range(WIDTH), range(HEIGHT), range(NUM_ACTIONS), range(WIDTH), range(HEIGHT)):
        P_[x, y, a, x_next, y_next] = get_probability_a(
            x, y, a, x_next, y_next, obstacles)

    P = P_.reshape(NUM_STATES, NUM_ACTIONS, WIDTH, HEIGHT).reshape(
        NUM_STATES, NUM_ACTIONS, NUM_STATES)

    NEGATIVE_REWARD = - 0.01
    POSITIVE_REWARD = 1

    R_ = np.zeros((WIDTH, HEIGHT, NUM_ACTIONS))

    R_ = R_ + NEGATIVE_REWARD
    R_[5, 3, 1] = POSITIVE_REWARD
    R_[6, 3, 4] = POSITIVE_REWARD
    R_[6, 4, 0] = POSITIVE_REWARD
    R_[6, 2, 2] = POSITIVE_REWARD

    R = R_.reshape(NUM_STATES, NUM_ACTIONS)

    print(P)
    print(R)

    mdp = MDP(P, R)

    T = WIDTH + HEIGHT + 1
    u = mdp.dynamic_programming(T)
    print(u.reshape(7, 6))
