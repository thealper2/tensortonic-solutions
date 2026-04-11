import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    """
    batch_size, H, W, n_features = features.shape
    output = np.zeros((batch_size, H, W, num_classes))
    return output
