import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    if isinstance(x, list):
        x = np.asarray(x)
    else:
        x = np.array([x])
        
    sigma = np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (np.exp(x) + 1)
    )
    return x * sigma
