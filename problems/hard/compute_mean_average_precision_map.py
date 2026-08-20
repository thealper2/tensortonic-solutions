import numpy as np

def average_precision(actual, predicted, k=None):
    """
    Compute Average Precision for a single query.
    """
    if not actual:
        return 0.0

    if k is not None:
        predicted = predicted[:k]

    total_relevant = len(actual)
    if total_relevant == 0:
        return 0.0

    num_hits = 0.0
    score = 0.0

    max_pos = len(predicted)
    for i in range(max_pos):
        p = predicted[i]
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / total_relevant

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_scores = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        sorted_indices = np.argsort(y_score)[::-1]
        relevant_indices = np.where(y_true == 1)[0]
        relevant_items = set(relevant_indices.tolist())
        predicted_items = sorted_indices.tolist()
        ap = average_precision(relevant_items, predicted_items, k)
        ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    return map_score, ap_scores