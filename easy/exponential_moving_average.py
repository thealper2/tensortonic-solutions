def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    n = len(values)
    ema = [values[0]]
    for i in range(1, n):
        mv = alpha * values[i] + (1 - alpha) * ema[i - 1]
        ema.append(mv)

    return ema
