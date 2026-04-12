import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        original_shape = x.shape
        if x.ndim == 1:
            x_flat = x.reshape(1, -1)
        elif x.ndim == 2:
            x_flat = x
        elif x.ndim == 3:
            batch = x.shape[0] if x.shape[0] != self.channels else 1
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            batch, c, h, w = x.shape
            x_flat = x.reshape(batch * h * w, c)
        
        out1 = x_flat @ self.W1.T
        out1 = relu(out1)
        out2 = out1 @ self.W2.T
        F_x = relu(out2)
        y_flat = F_x + x_flat
        y = y_flat.reshape(original_shape)
        return y
