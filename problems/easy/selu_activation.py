import numpy as np

def selu(x):
    """
    Apply SELU activation to each element.
    """
    alpha = 1.6733
    lambda_ = 1.0507
    x = np.array(x)
    z = lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return z.tolist()
