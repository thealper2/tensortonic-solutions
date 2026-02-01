import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    np_x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-np_x))
