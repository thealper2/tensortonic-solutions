import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.atleast_1d(w)
    m = np.atleast_1d(m)
    v = np.atleast_1d(v)
    grad = np.atleast_1d(grad)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    nesterov_term = beta1 * m_new + (1 - beta1) * grad
    w_new = w - lr * nesterov_term / (np.sqrt(v_new) + eps)
    return w_new, m_new, v_new
