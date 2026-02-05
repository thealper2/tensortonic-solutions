import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    w = np.array(w)
    grad = np.array(grad)
    E_grad_sq = np.array(E_grad_sq)
    E_update_sq = np.array(E_update_sq)
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * (grad ** 2)
    dx = -np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_new + eps) * grad
    E_update_sq_new = rho * E_update_sq + (1 - rho) * (dx ** 2)
    w_new = w + dx
    return w_new, E_grad_sq_new, E_update_sq_new
