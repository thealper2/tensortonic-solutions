import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)"""
    z = np.concatenate([r_t * h_prev, x_t], axis=-1)
    h_t = np.tanh(np.dot(z, W_h.T) + b_h)
    return h_t
