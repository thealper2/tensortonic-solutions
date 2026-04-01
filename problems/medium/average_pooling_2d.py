import numpy as np

def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    X = np.array(X)
    n_H, n_W = X.shape
    out_height = (n_H - pool_size) // pool_size + 1
    out_width = (n_W - pool_size) // pool_size + 1
    pooled_output = np.zeros((out_height, out_width))

    for h in range(out_height):
        for w in range(out_width):
            h_start = h * pool_size
            h_end = h_start + pool_size
            w_start = w * pool_size
            w_end = w_start + pool_size
            X_slice = X[h_start:h_end, w_start:w_end]
            pooled_output[h, w] = np.mean(X_slice)

    return pooled_output.tolist()
