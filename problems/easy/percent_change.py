def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    n = len(series)
    result = []
    for i in range(1, n):
        if series[i - 1] == 0:
            pct = 0.0
        else:
            pct = (series[i] - series[i - 1]) / series[i - 1]
            
        result.append(pct)

    return result
