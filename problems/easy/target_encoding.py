def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    sums = {}
    counts = {}

    for c, t in zip(categories, targets):
        sums[c] = sums.get(c, 0.0) + t
        counts[c] = counts.get(c, 0) + 1

    means = {c: sums[c] / counts[c] for c in sums}
    return [means[c] for c in categories]
