import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    x = np.array(x_T, dtype=np.float64)
    betas = np.array(betas, dtype=np.float64)
    epsilon_preds = np.array(epsilon_preds, dtype=np.float64)
    z_values = np.array(z_values, dtype=np.float64)

    T = len(betas)

    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    for t in range(T, 0, -1):
        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        epsilon_pred = epsilon_preds[T - t]

        mu = (
            x
            - (beta_t / np.sqrt(1 - alpha_bar_t)) * epsilon_pred
        ) / np.sqrt(alpha_t)

        if t > 1:
            z = z_values[T - t]
            sigma_t = np.sqrt(beta_t)
            x = mu + sigma_t * z
        else:
            x = mu

    return np.round(x, 4)