import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        y = y.reshape(1, -1)

    distances = np.linalg.norm(a - b, axis=1)
    losses = y * (distances ** 2) + (1 - y) * np.maximum(0, margin - distances) ** 2
    if reduction == "mean":
        result = np.mean(losses)
    else:
        result = np.sum(losses)

    return result
