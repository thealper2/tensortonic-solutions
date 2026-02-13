import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return None

    N, D = X.shape
    if N < 2:
        return None

    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (N - 1)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / denom

    zero_var = std == 0
    corr[zero_var, :] = np.nan
    corr[:, zero_var] = np.nan
    return corr
