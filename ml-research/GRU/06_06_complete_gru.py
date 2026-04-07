import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last)."""
        batch_size, seq_len, input_dim = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim))
        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            z = np.concatenate([h_t, x_t], axis=1)
            r_t = sigmoid(np.dot(z, self.W_r.T) + self.b_r)
            z_t = sigmoid(np.dot(z, self.W_z.T) + self.b_z)
            z = np.concatenate([r_t * h_t, x_t], axis=1)
            h_tilde_t = np.tanh(np.dot(z, self.W_h.T) + self.b_h)
            h_t = z_t * h_tilde_t + (1 - z_t) * h_t
            y_t = np.dot(h_t, self.W_y.T) + self.b_y
            outputs.append(y_t)

        y = np.stack(outputs, axis=1)
        return y, h_t
