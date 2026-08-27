import numpy as np

def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)

    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n_samples
    parent_gini = 1 - np.sum(probs ** 2)

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feature in range(n_features):
        values = X[:, feature]
        unique_values = np.unique(values)

        if len(unique_values) <= 1:
            continue

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_y = y[left_mask]
            left_counts = np.bincount(left_y, minlength=len(classes))
            left_probs = left_counts / len(left_y)
            left_gini = 1 - np.sum(left_probs ** 2)

            right_y = y[right_mask]
            right_counts = np.bincount(right_y, minlength=len(classes))
            right_probs = right_counts / len(right_y)
            right_gini = 1 - np.sum(right_probs ** 2)

            split_gini = (len(left_y) / n_samples) * left_gini + (len(right_y) / n_samples) * right_gini

            gain = parent_gini - split_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
            elif gain == best_gain and feature == best_feature and threshold < best_threshold:
                best_threshold = threshold

    return [int(best_feature), float(best_threshold)]