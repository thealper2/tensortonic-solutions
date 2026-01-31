import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    if len(x) != len(y):
        raise ValueError("Mismatched Lengths")
        
    dot_p = sum([i * j for i, j in zip(x, y)])
    return float(dot_p)
