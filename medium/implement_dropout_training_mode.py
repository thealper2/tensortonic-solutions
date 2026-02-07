import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    rand = rng.random(x.shape) if rng else np.random.random(x.shape)
    keep_prob = 1 - p
    dropout_pattern = (rand < keep_prob).astype(x.dtype) / keep_prob
    output = x * dropout_pattern
    return output, dropout_pattern
