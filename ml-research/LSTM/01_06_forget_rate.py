import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(
    h_prev: np.ndarray, 
    x_t: np.ndarray,
    W_f: np.ndarray, 
    b_f: np.ndarray
) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    if h_prev.ndim == 1 and x_t.ndim == 2:
        h_prev = h_prev[np.newaxis, :]
    elif h_prev.ndim == 2 and x_t.ndim == 1:
        x_t = np.tile(x_t[np.newaxis, :], (h_prev.shape[0], 1))
    elif h_prev.ndim == 1 and x_t.ndim == 1:
        concat = np.concatenate([h_prev, x_t], axis=0)
        f_t = sigmoid(np.dot(W_f, concat) + b_f)
        return f_t
    
    if h_prev.ndim == 1:
        h_prev = h_prev[np.newaxis, :]
    if x_t.ndim == 1:
        x_t = x_t[np.newaxis, :]
    
    concat = np.concatenate([h_prev, x_t], axis=-1)
    f_t = sigmoid(np.dot(concat, W_f.T) + b_f)
    
    return f_t
