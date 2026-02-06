import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    _, n_features = X.shape
    I = np.eye(n_features)
    XtX = X.T @ X
    regularized = XtX + lam *I
    XtY = X.T @ y
    w = np.linalg.inv(regularized) @ XtY
    return w.tolist()
