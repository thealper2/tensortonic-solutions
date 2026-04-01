def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    H = len(image)
    W = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])

    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded = [[0.0 for _ in range(padded_W)] for _ in range(padded_H)]

    for i in range(H):
        for j in range(W):
            padded[i + padding][j + padding]= image[i][j]

    out_h = (padded_H - kh) // stride + 1
    out_w = (padded_W - kw) // stride + 1

    output = [[0.0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for m in range(kh):
                for n in range(kw):
                    s += padded[i * stride + m][j * stride + n] * kernel[m][n]

            output[i][j] = s

    return output
