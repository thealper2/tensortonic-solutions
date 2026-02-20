import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    N, C_in, H, W_in = x.shape
    C_out, C_in_w, KH, KW = W.shape
    assert C_in == C_in_w, "Channel mismatch"
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    y = np.zeros((N, C_out, H_out, W_out), dtype=float)
    for u in range(KH):
        for v in range(KW):
            x_slice = x[:, :, u:u+H_out, v:v+W_out]
            y += np.einsum('nchw,oc->nohw', x_slice, W[:, :, u, v])

    y += b[None, :, None, None]
    return y
