import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    sq_x = np.square(x)
    pad = n // 2
    sq_padded = np.pad(sq_x, ((0, 0), (pad, pad), (0, 0), (0, 0)), mode='constant')
    channel_sum = np.zeros_like(x)
    for i in range(n):
        channel_sum += sq_padded[:, i:i+x.shape[1], :, :]

    scale = (k + alpha * channel_sum) ** beta
    output = x / scale
    return output
