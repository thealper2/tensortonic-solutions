import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t, h_prev, h_t, W, U, b = cache
    h_t = np.array(h_t)
    dz = dh * (1 - h_t**2)
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz.copy()
    dx_t = dz @ W
    dh_prev = dz @ U
    return dx_t, dh_prev, dW, dU, db