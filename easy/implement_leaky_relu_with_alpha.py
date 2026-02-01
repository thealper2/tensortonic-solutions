import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x)
    z = np.where(
        x < 0,
        alpha * x,
        x
    )
    return z
