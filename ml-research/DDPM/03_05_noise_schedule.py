import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    linear_beta = np.linspace(beta_1, beta_T, T)
    return linear_beta

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    steps = np.arange(T + 1)
    f_t = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas_bar = np.clip(alphas_bar, 0.0001, 0.9999)
    alphas_bar = alphas_bar[1:]
    return alphas_bar

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    T = len(alpha_bars)
    betas = np.zeros(T)
    alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
    alphas = alpha_bars / alpha_bars_prev
    betas = 1 - alphas
    betas = np.clip(betas, 0.0001, 0.9999)
    return betas