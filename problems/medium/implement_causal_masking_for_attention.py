import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.array(scores)
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T)), k=1)
    broadcast_shape = (1,) * (scores.ndim - 2) + (T, T)
    mask_broadcast = mask.reshape(broadcast_shape)
    logits = np.where(mask_broadcast == 1, mask_value, scores)
    return logits