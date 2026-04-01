import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    preds = []
    for i, pred in enumerate(predictions):
        if targets[i] == 1:
            preds.append(pred)
        else:
            preds.append(1 - pred)

    loss = 0.0
    for pred in preds:
        loss += - alpha * ((1 - pred) ** gamma) * math.log(pred)

    return loss / len(preds)
