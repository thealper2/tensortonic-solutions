def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    n = len(X)
    d_in = len(X[0])
    d_out = len(W[0])

    Y = [[0.0] * d_out for _ in range(n)]
    for i in range(n):
        for j in range(d_out):
            dot_product = 0.0
            for k in range(d_in):
                dot_product += X[i][k] * W[k][j]

            Y[i][j] = dot_product + b[j]

    return Y
