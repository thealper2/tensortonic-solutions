def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series)

    if variance == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for lag in range(max_lag + 1):
        if lag == 0:
            result.append(1.0)
        else:
            cov = 0.0
            for t in range(n - lag):
                cov += (series[t] - mean) * (series[t + lag] - mean)

            result.append(cov / variance)

    return result