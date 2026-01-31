import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.array(v)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm_safe = np.where(norm > 1e-10, norm, 1.0)
    return np.where(norm > 1e-10, v / norm_safe, v)
