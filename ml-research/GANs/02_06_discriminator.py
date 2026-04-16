import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    logits = np.dot(x, W)
    sigmoid = 1 / (1 + np.exp(-logits))
    return sigmoid
