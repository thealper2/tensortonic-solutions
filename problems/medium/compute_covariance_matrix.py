import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return None

    N, D = X.shape
    if N < 2:
        return None

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov_mat = (1 / (N - 1)) * X_centered.T @ X_centered
    return cov_mat
