def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    n = len(values)
    if n == 0:
        return []

    if n == 1:
        return [0.0]

    sorted_vals = sorted(values)

    def median(arr):
        m = len(arr)
        if m % 2 == 1:
            return float(arr[m // 2])
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2.0

    med = median(sorted_vals)

    if n % 2 == 1:
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2 + 1:]
    else:
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2:]

    q1 = median(lower)
    q3 = median(upper)
    iqr = q3 - q1

    if iqr == 0:
        return [float(x - med) for x in values]
    else:
        return [float((x - med) / iqr) for x in values]
