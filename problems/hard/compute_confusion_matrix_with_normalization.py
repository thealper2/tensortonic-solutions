import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.zeros((0, 0))

    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    if np.max(y_true) >= num_classes or np.max(y_pred) >= num_classes:
        raise ValueError("Labels out of range")

    if np.min(y_true) < 0 or np.min(y_pred) < 0:
        raise ValueError("Labels must be non-negative")

    indices = y_true * num_classes + y_pred
    counts = np.bincount(indices, minlength=num_classes * num_classes)
    cm = counts.reshape(num_classes, num_classes).astype(np.float64)

    if normalize == 'none':
        return cm.astype(np.int64)

    elif normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cm / row_sums

    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return cm / col_sums

    elif normalize == 'all':
        total = cm.sum()
        if total == 0:
            return cm

        return cm / total

    else:
        raise ValueError(f"Invalid normalization mode: {normalize}")