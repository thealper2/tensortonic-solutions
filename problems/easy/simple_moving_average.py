def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    result = []
    n = len(values)
    for i in range(n - window_size + 1):
        sub  = values[i:i+window_size]
        moving_average = sum(sub) / window_size
        result.append(moving_average)

    return result
