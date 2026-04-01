import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    loss = 0
    for p_, q_ in zip(p, q):
        if p_ == 0:
            continue

        loss += p_ * np.log(p_ / (q_ + eps))
    
    return loss
