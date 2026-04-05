import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    batch_size, n_H, n_W, n_C = x.shape
    out_height = n_H // stride
    out_width = n_W // stride
    output =  np.zeros((batch_size, out_height, out_width, n_C))
    return output
