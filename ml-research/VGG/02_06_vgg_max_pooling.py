import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    x = np.asarray(x)
    batch_size, n_H, n_W, n_C = x.shape
    stride = 2
    pool_size = 2
    
    out_height = n_H // stride
    out_width = n_W // stride
    pooled_output = np.zeros((batch_size, out_height, out_width, n_C))

    for b in range(batch_size):
        for c in range(n_C):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    x_slice = x[b, h_start:h_end, w_start:w_end, c]
                    pooled_output[b, h, w, c] = np.max(x_slice)

    return pooled_output
