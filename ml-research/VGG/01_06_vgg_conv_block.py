import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """
    batch_size, height, width, in_channels = x.shape
    current = x

    for i in range(num_convs):
        if i == 0:
            conv_in_channels = in_channels
        else:
            conv_in_channels = out_channels

        weights = np.random.randn(3, 3, conv_in_channels, out_channels) * 0.01
        biases = np.zeros(out_channels)
        padded = np.pad(current, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        conv_output = np.zeros((batch_size, height, width, out_channels))
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    for out_c in range(out_channels):
                        region = padded[b, h:h+3, w:w+3, :]
                        conv_output[b, h, w, out_c] = np.sum(region * weights[:, :, :, out_c]) + biases[out_c]

        current = np.maximum(0, conv_output)

    return current
