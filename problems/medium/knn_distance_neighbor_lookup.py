import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    diff = X_test[:, None, :] - X_train[None, :, :]
    distances = np.sum(diff ** 2, axis=2)

    sorted_indices = np.argsort(distances, axis=1)
    
    result = sorted_indices[:, :k]

    if k > n_train:
        pad_width = k - n_train
        pad = -np.ones((n_test, pad_width), dtype=np.int64)
        result = np.concatenate([result, pad], axis=1)

    return result