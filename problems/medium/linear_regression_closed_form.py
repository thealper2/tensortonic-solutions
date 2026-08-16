import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X)
    y = np.array(y)
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)
    theta = np.linalg.solve(X_transpose_X, X_transpose_y)
    theta = np.round(theta, 4).flatten().tolist()
    return theta