import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    norms = [1.0]
    current_norm = 1.0
    for t in range(1, T + 1):
        current_norm *= spectral_norm
        norms.append(current_norm)

    return norms
