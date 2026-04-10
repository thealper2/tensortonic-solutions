import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        batch_size, seq_len, _ = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim))
        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            h_t = np.tanh(
                x_t @ self.W_xh.T +
                h_t @ self.W_hh.T +
                self.b_h
            )
            h_states.append(h_t)

        h_all = np.stack(h_states, axis=1)

        batch_size, seq_len, hidden_dim = h_all.shape
        h_flat = h_all.reshape(-1, hidden_dim)
        y_flat = h_flat @ self.W_hy.T + self.b_y
        Y = y_flat.reshape(batch_size, seq_len, self.output_dim)
        h_final = h_states[-1]
        return Y, h_final
