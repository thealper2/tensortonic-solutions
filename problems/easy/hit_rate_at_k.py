def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    if not recommendations:
        return 0.0

    hits = 0
    num_users = len(recommendations)

    for recs, rel in zip(recommendations, ground_truth):
        top_k = set(recs[:k])
        relevant = set(rel)
        if top_k & relevant:
            hits += 1

    return hits / num_users
