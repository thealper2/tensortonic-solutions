import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.array(betas)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar = np.round(alpha_bar, 4)
    return alpha_bar.tolist()

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    betas = np.array(betas)
    x_0 = np.array(x_0)
    epsilon = np.array(epsilon)
    
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_t = alpha_bar[t - 1]
    sqrt_alpha_bar = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
    x_t = np.round(x_t, 4)
    return x_t.tolist()