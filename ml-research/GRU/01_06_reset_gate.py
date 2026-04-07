import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)"""
    z = np.concatenate([h_prev, x_t], axis=-1)
    r_t = sigmoid(np.dot(z, W_r.T) + b_r)
    return r_t
