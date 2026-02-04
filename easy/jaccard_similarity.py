def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_a = set(set_a)
    set_b = set(set_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if not union:
        return 0.0

    return intersection / union
