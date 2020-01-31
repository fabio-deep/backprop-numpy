import numpy as np

def binary_cross_entropy(y_hat, y):
    cost = np.mean(-y * np.log(y_hat) - (1. - y) * np.log(1. - y_hat))
    return cost

def categorical_cross_entropy(y_hat, y):
    cost = np.mean(-np.sum(y * np.log(y_hat), 0))
    return cost
