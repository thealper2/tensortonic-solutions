def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    hits = sum([1 for v in relevant if v in top_k])
    precision_at_k = hits / k
    recall_at_k = hits / len(relevant)
    return [precision_at_k, recall_at_k]
