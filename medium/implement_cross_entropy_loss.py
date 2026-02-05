import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct_class_probs = y_pred[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(correct_class_probs))
