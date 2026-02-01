import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    n = len(X_test)
    values, counts = np.unique(y_train, return_counts=True)
    idx = np.argmax(counts)
    majority = values[idx]
    return np.full((n,), majority)
