def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    limit = (6 / fan_in) ** 0.5
    W_init = [[w * 2 * limit - limit for w in row] for row in W]
    return W_init
