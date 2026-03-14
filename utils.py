import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))