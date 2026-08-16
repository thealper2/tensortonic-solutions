import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    result = {}
    for feature in train_dist.keys():
        train_pct = np.array(train_dist[feature], dtype=np.float64) + eps
        serve_pct = np.array(serving_dist[feature], dtype=np.float64) + eps

        psi_bins = (serve_pct - train_pct) * np.log(serve_pct / train_pct)
        psi = np.sum(psi_bins)

        result[feature] = {
            "psi": float(psi),
            "skewed": bool(psi >= threshold),
        }
        
    return result