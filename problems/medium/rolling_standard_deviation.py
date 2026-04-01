import numpy as np

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    result = []
    n = len(values)
    values = np.asarray(values)
    for i in range(n - window_size + 1):
        sub = values[i:i+window_size]
        mean = np.mean(sub)
        std = np.sqrt(np.sum((sub - mean) ** 2) / window_size)
        result.append(std)

    return result
