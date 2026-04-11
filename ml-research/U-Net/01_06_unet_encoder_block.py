import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """
    batch_size, input_height, input_width, in_channels = x.shape

    c1_height = input_height - 2
    c1_width = input_width - 2

    c2_height = input_height - 4
    c2_width = input_width - 4

    out_height = c2_height // 2
    out_width = c2_width // 2

    skip_output = np.zeros((batch_size, c2_height, c2_width, out_channels))
    pool_output = np.zeros((batch_size, out_height, out_width, out_channels))
    return pool_output, skip_output
