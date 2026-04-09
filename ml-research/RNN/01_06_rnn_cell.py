import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    z1 = np.dot(h_prev, W_hh.T)
    z2 = np.dot(x_t, W_xh.T)
    h_t = np.tanh(z1 + z2 + b_h)
    return h_t
