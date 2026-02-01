def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    limit = (6 / (fan_in + fan_out)) ** 0.5
    W_init = [[w * 2 * limit - limit for w in row] for row in W]
    return W_init
