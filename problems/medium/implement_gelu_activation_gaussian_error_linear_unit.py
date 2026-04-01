import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
