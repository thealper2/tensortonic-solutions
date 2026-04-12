import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    g = x.copy().astype(float).reshape(-1, 1)
    for J in reversed(gradients_F):
        I = np.eye(J.shape[0], dtype=float)
        g = (I + J) @ g

    return g.flatten()

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    g = x.copy().astype(float).reshape(-1, 1)
    for J in reversed(gradients_F):
        g = J @ g

    return g.flatten()
