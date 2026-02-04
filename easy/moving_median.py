def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    n = len(values)
    medians = []

    for i in range(n - window_size + 1):
        window = sorted(values[i:i+window_size])
        k = window_size

        if k % 2 == 1:
            median = float(window[k // 2])
        else:
            median = (window[k // 2 - 1] + window[k // 2]) / 2.0

        medians.append(median)

    return medians
