import numpy as np


def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities = np.asarray(priorities, dtype=float)
    N = priorities.shape[0]
    powered = priorities ** alpha
    total = powered.sum()
    probs = powered / total
    raw_weights = (N * probs) ** (-beta)
    weights = raw_weights / raw_weights.max()
    return [probs.tolist(), weights.tolist()]
