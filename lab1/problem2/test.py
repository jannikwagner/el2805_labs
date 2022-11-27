import numpy as np
import itertools

eta = np.array([[0, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [0, 2], ])

p = 2
eta = np.array(list(itertools.product(range(p+1), range(p+1))))


def get_phi(eta, s):
    return np.cos(np.pi * eta @ s)


def get_Q(w, eta, s):
    return w @ get_phi(eta, s)


def get_Q_a(w, eta, s, a):
    return w[a] @ get_phi(eta, s)
