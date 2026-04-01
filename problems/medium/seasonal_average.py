def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    n = len(series)
    result = []
    for p in range(period):
        values = [series[i] for i in range(p, n, period)]
        if values:
            result.append(sum(values) / len(values))
        else:
            result.append(0.0)

    return result
