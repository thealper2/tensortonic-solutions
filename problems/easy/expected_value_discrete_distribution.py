import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    if 1 - sum(p) > 1e-6:
        raise ValueError()

    prod = np.dot(x, p)
    return prod
