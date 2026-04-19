import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    x_t = np.array(x_t)
    epsilon_pred = np.array(epsilon_pred)
    betas = np.array(betas)
    
    beta_t = betas[t - 1]
    alpha_t = 1 - beta_t

    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_t = alpha_bar[t - 1]

    sqrt_alpha_t = np.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar_t)

    mean_coeff = 1.0 / sqrt_alpha_t
    noise_removal_coeff = beta_t / sqrt_one_minus_alpha_bar

    mu = mean_coeff * (x_t - noise_removal_coeff * epsilon_pred)

    if t > 1:
        sigma_t = np.sqrt(beta_t)
        if z is None:
            z = np.random.randn(*x_t.shape)
        else:
            z = np.array(z)

        x_prev = mu + sigma_t * z
    else:
        x_prev = mu

    return x_prev