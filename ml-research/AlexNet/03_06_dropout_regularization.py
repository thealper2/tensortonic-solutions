import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    if not training:
        return x

    rand = np.random.random(x.shape)
    keep_prob = 1 - p
    dropout_pattern = (rand < keep_prob).astype(x.dtype) / keep_prob
    output = x * dropout_pattern
    return output
