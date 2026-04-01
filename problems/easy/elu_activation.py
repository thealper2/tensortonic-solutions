import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.atleast_1d(x)
    z = np.where(
        x <= 0,
        alpha * (np.exp(x) - 1),
        x
    )
    return z.tolist()
