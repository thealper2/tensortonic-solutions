def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    candidates = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in candidates[:k]]
