import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_norm + beta

    elif x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        out = gamma * x_norm + beta
        
    return out
