import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    if h_prev.ndim == 1:
        h_prev = h_prev.reshape(1, -1)
    if x_t.ndim == 1:
        x_t = x_t.reshape(1, -1)
    
    if h_prev.shape[0] != x_t.shape[0]:
        batch_size = max(h_prev.shape[0], x_t.shape[0])
        if h_prev.shape[0] == 1:
            h_prev = np.repeat(h_prev, batch_size, axis=0)
        if x_t.shape[0] == 1:
            x_t = np.repeat(x_t, batch_size, axis=0)
    
    concat = np.concatenate([h_prev, x_t], axis=1)
    i_t = sigmoid(np.dot(concat, W_i.T) + b_i)
    c_tilde = np.tanh(np.dot(concat, W_c.T) + b_c)
    return i_t, c_tilde
