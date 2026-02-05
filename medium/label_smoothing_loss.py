import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)
    smoothed = []
    for i, pred in enumerate(predictions):
        if i == target:
            q_i = (1 - epsilon) + epsilon / K
        else:
            q_i = epsilon / K

        smoothed.append(q_i)

    loss = 0.0
    for pred, q_i in zip(predictions, smoothed):
        loss += -q_i * math.log(pred)

    return loss
