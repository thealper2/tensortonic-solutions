import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.atleast_1d(x)
    z = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return z
