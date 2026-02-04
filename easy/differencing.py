def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    for _ in range(order):
        new_series = []
        n = len(series)
        for i in range(n - 1):
            new_series.append(series[i + 1] - series[i])

        series = new_series

    return series
