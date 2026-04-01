def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0] * len(values)

    bin_width = (max_val - min_val) / num_bins
    bins = []

    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, num_bins - 1)
        bins.append(bin_idx)

    return bins
