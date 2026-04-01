def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    k = len(weights)
    n = len(values)
    w_sum = sum(weights)
    result = []

    for i in range(n - k + 1):
        weighted_sum = 0.0
        for j in range(k):
            weighted_sum += weights[j] * values[i + j]

        result.append(weighted_sum / w_sum)

    return result
