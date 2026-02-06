import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)
    if max_norm <= 0:
        return g
        
    norm = np.linalg.norm(g)
    if norm > max_norm:
        scaled = g * (max_norm / norm)
    else:
        scaled = g.copy()
    
    return scaled
