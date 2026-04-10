import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size, seq_len, input_dim = X.shape
    hidden_dim = h_0.shape[1]

    h_states = []
    h_t = h_0.copy()

    for t in range(seq_len):
        x_t = X[:, t, :]
        h_t = np.tanh(
            np.dot(h_t, W_hh.T) +
            np.dot(x_t, W_xh.T) +
            b_h
        )

        h_states.append(h_t)

    h_all = np.stack(h_states, axis=1)
    h_final = h_states[-1]
    return h_all, h_final
