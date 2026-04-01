import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        "min": np.full(D, np.inf, dtype=float),
        "max": np.full(D, -np.inf, dtype=float),
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X = np.asarray(X_batch, dtype=float)
    batch_min = X.min(axis=0)
    batch_max = X.max(axis=0)
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)
    denom = np.maximum(state["max"] - state["min"], eps)
    X_norm = (X - state["min"]) / denom
    return X_norm
