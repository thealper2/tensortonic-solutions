import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    S = np.dot(Z1, Z2.T) / temperature
    S_max = np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum_exp = np.sum(S_exp, axis=1)
    diag_exp = np.diag(S_exp)
    L = -np.log(diag_exp / S_sum_exp)
    return np.mean(L)