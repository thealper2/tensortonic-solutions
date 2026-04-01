def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    normalized = []

    for row in matrix:
        rated = [v for v in row if v != 0]
        if not rated:
            normalized.append([0.0] * len(row))
            continue

        mean = sum(rated) / len(rated)
        normalized.append([v - mean if v != 0 else 0.0 for v in row])

    return normalized
