import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    epsilon = np.array(epsilon)
    epsilon_pred = np.array(epsilon_pred)
    squared_diff = (epsilon - epsilon_pred) ** 2
    mse_loss = np.mean(squared_diff)
    return float(mse_loss)