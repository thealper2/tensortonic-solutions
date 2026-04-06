import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(
    h_prev: np.ndarray, 
    x_t: np.ndarray, 
    C_t: np.ndarray,
    W_o: np.ndarray, 
    b_o: np.ndarray
) -> tuple:
    """Compute output gate and hidden state."""
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
    o_t = sigmoid(np.dot(concat, W_o.T) + b_o)
    h_t = o_t * np.tanh(C_t)
    return o_t, h_t
