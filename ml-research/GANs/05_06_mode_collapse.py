import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    std_devs = np.std(generated_samples, axis=0)
    diversity_score = np.mean(std_devs)
    is_collapsed = diversity_score < threshold
    return { "diversity_score": diversity_score, "is_collapsed": is_collapsed }
