import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.lexsort((1 - y_true, -y_score))
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_true_sorted) - 1]
    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = y_score_sorted[threshold_idxs]
    tpr = tps / P if P > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N > 0 else np.zeros_like(fps, dtype=float)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds
