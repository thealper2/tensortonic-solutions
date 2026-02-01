import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)

    new_G = G + g ** 2
    new_w = w - (lr / (np.sqrt(new_G + eps))) * g
    return new_w, new_G
