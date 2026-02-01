import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    loss = np.maximum(0, margin - y_true * y_score)
    if reduction == "mean":
        return np.mean(loss)
    else:
        return np.sum(loss)
