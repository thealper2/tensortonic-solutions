import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    var = np.exp(log_var)
    kl_per_sample = -0.5 * np.sum(1 + log_var - (mu ** 2) - var, axis=1)
    kl = np.mean(kl_per_sample)
    return float(kl)
