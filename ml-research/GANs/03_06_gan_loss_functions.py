import numpy as np

def clip_probs(probs):
    return np.clip(probs, 1e-8, 1-1e-8)

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    disc_loss = -np.mean(np.log(clip_probs(real_probs)) + np.log(1 - clip_probs(fake_probs)))
    return disc_loss

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    gen_loss = -np.mean(np.log(clip_probs(fake_probs)))
    return gen_loss
