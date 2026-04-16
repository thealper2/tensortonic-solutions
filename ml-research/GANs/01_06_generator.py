import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    g_z = np.tanh(np.dot(z, W) + b)
    return g_z
