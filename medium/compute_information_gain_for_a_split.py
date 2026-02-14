import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)
    N = len(y)
    y_left = y[mask]
    y_right = y[~mask]
    nL = len(y_left)
    nR = len(y_right)
    if nL == 0 or nR == 0:
        return 0.0

    H_parent = _entropy(y)
    H_children = (nL / N) * _entropy(y_left) + (nR / N) * _entropy(y_right)
    IG = H_parent - H_children
    return float(IG)
