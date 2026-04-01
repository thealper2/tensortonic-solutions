def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return float(tp / n)
