def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    n = len(values)
    ranks = [0.0] * n
    sorted_indices = sorted(range(n), key=lambda i: values[i])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[sorted_indices[j]] == values[sorted_indices[j + 1]]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank

        i = j + 1

    return ranks
