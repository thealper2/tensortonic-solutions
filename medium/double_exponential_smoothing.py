def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    n = len(series)
    if n == 0:
        return []
    
    if n == 1:
        return [float(series[0])]

    level = float(series[0])
    trend = float(series[1] - series[0])

    levels = [level]

    for t in range(1, n):
        prev_level = level
        level = alpha * series[t] + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        levels.append(level)

    return levels
