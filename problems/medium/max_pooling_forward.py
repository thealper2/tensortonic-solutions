import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    n_H, n_W = X.shape
    out_height = (n_H - pool_size) // stride + 1
    out_width = (n_W - pool_size) // stride + 1
    pooled_output = np.zeros((out_height, out_width))

    for h in range(out_height):
        for w in range(out_width):
            h_start = h * stride
            h_end = h_start + pool_size
            w_start = w * stride
            w_end = w_start + pool_size
            X_slice = X[h_start:h_end, w_start:w_end]
            pooled_output[h, w] = np.max(X_slice)

    return pooled_output.tolist()
