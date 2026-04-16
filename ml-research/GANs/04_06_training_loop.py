import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clip_probs(probs):
    return np.clip(probs, 1e-8, 1-1e-8)

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    r_logits = np.dot(real_data, D_W)
    f_logits = np.dot(fake_data, D_W)
    r_probs = sigmoid(r_logits)
    f_probs = sigmoid(f_logits)
    d_loss = -np.mean(np.log(clip_probs(r_probs)) + np.log(1 - clip_probs(f_probs)))
    g_loss = -np.mean(np.log(clip_probs(f_probs)))
    return { "d_loss": d_loss, "g_loss": g_loss }
