import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        if len(x.shape) == 4:
            N, C, H, W = x.shape
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_flat = x

        if training:
            mean = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x_flat - mean) / np.sqrt(var + self.eps)
        x_scaled = x_norm * self.gamma + self.beta
        if len(x.shape) == 4:
            x_scaled = x_scaled.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return x_scaled

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    residual = x @ W1
    residual = bn1.forward(residual, training=True)
    residual = relu(residual)

    residual = residual @ W2
    residual = bn2.forward(residual, training=True)

    out = residual + x
    out = relu(out)
    return out


def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    residual = bn1.forward(x, training=True)
    residual = relu(residual)
    residual = residual @ W1

    residual = bn2.forward(residual, training=True)
    residual = relu(residual)
    residual = residual @ W2

    out = residual + x
    return out

