import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    mse = np.sum((x - x_recon) ** 2, axis=1).mean()
    kl_divergence = (-0.5 * np.sum(1 + log_var - (mu ** 2) - np.exp(log_var), axis=1)).mean()
    total = mse + kl_divergence
    return { "total": total, "recon": mse, "kl": kl_divergence }
