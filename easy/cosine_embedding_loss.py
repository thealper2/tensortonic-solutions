import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.array(x1)
    x2 = np.array(x2)

    dot_product = np.dot(x1, x2)
    norm_A = np.linalg.norm(x1)
    norm_B = np.linalg.norm(x2)
    cosine_sim = dot_product / (norm_A * norm_B)
    if label == 1:
        loss = 1 - cosine_sim
    else:
        loss = max(0, cosine_sim - margin)

    return loss
