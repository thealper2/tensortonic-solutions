import numpy as np

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    rel = np.asarray(relevance_scores, dtype=float)

    if rel.size == 0:
        return 0.0

    k = min(k, len(rel))

    positions = np.arange(1, k + 1)
    gains = 2 ** rel[:k] - 1
    discounts = np.log2(positions + 1)
    dcg = np.sum(gains / discounts)

    ideal_rel = np.sort(rel)[::-1][:k]
    ideal_gains = 2 ** ideal_rel - 1
    idcg = np.sum(ideal_gains / discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg
