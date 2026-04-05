import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    batch_size, input_height, input_width, in_channels = image.shape
    kernel_height, kernel_width = 11, 11
    stride = 4
    out_channels = 96
    
    output_height = (input_height - kernel_height + 4) // stride + 1
    output_width = (input_width - kernel_width + 4) // stride + 1
    output_matrix = np.zeros((batch_size, output_height, output_width, out_channels))

    return output_matrix
