import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x_2d, x_was_1d = _as2d(x, x.shape[-1] if hasattr(x, 'shape') else len(x))
    h_prev_2d, h_was_1d = _as2d(h_prev, h_prev.shape[-1] if hasattr(h_prev, 'shape') else len(h_prev))

    if x_2d.shape[0] != h_prev_2d.shape[0]:
        raise ValueError(f"Batch size mismatch: x has {x_2d.shpae[0]}, h_prev has {h_prev_2d.shape[0]}")

    Wz = params["Wz"]
    Uz = params["Uz"]
    bz = params["bz"]
    Wr = params["Wr"]
    Ur = params["Ur"]
    br = params["br"]
    Wh = params["Wh"]
    Uh = params["Uh"]
    bh = params["bh"]

    z_t = _sigmoid(x_2d @ Wz + h_prev_2d @ Uz + bz)
    r_t = _sigmoid(x_2d @ Wr + h_prev_2d @ Ur + br)
    h_tilde = np.tanh(x_2d @ Wh + (r_t * h_prev) @ Uh + bh)
    h_next_2d = (1 - z_t) * h_prev + z_t * h_tilde

    if x_was_1d and h_was_1d:
        return h_next_2d[0]
    
    return h_next_2d