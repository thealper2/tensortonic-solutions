import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.array(p)
    y = np.array(y)
    p = p.flatten()
    y = y.flatten()
    intersection = np.sum(p * y)
    dice_score = (2.0 * intersection + eps) / (np.sum(y) + np.sum(p) + eps)
    loss = 1 - dice_score
    return loss
