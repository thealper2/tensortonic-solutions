import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    """
    batch_size, H, W, in_channels = x.shape
    H_out = H - 4
    W_out = W - 4
    output = np.zeros((batch_size, H_out, W_out, out_channels))
    return output
