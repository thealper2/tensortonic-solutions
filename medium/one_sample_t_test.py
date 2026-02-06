import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    n = len(x)
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.sqrt(np.sum((x - mean) ** 2) / (n - 1))
    t = (mean - mu0) / (std / np.sqrt(n))
    return float(t)
