def recall_at_k(retrieved_ids, expected_ids, k):
    """
    Returns: fraction of expected evidence found in the first k results
    """
    retrieved_subset = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    intersection = retrieved_subset.intersection(expected_set)
    return len(intersection) / len(expected_set)