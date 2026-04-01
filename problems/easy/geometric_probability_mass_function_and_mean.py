import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.asarray(k, dtype=int)
    pmf = (1.0 - p) ** (k - 1) * p
    mean = float(1.0 / p)
    return pmf, mean
