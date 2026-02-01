import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.atleast_1d(x)
    z = np.where(
        x < 0,
        0,
        x
    )
    return z
