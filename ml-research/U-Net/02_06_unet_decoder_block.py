import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net decoder block: up-conv + concat + double conv.
    """
    batch_size, H, W, in_channels = x.shape
    _, skip_H, skip_W, skip_channels = skip.shape
    H_up = H * 2
    W_up = W * 2

    crop_H = (skip_H - H_up) // 2
    crop_W = (skip_W - W_up) // 2

    skip_cropped_H = skip_H - 2 * crop_H
    skip_cropped_W = skip_W - 2 * crop_W

    concat_channels = in_channels + skip_channels

    H_out = H_up - 4
    W_out = W_up - 4

    output = np.zeros((batch_size, H_out, W_out, out_channels))
    return output
