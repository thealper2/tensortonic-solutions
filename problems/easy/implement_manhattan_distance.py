import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    distance = np.sum(np.abs(x - y))
    return float(distance)
