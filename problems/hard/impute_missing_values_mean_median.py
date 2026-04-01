import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X)
    X_imputed = X.copy()
    if X_imputed.ndim == 1:
        mask = np.isnan(X_imputed)
        valid = ~mask
        if np.any(valid):
            if strategy == "mean":
                stat = np.mean(X_imputed[valid])
            else:
                stat = np.median(X_imputed[valid])
        else:
            stat = 0.0

        X_imputed[mask] = stat
        return X_imputed
    
    N, D = X_imputed.shape
    for j in range(D):
        col = X_imputed[:, j]
        mask = np.isnan(col)
        valid = ~mask
        if np.any(valid):
            if strategy == "mean":
                stat = np.mean(col[valid])
            else:
                stat = np.median(col[valid])

        else:
            stat = 0.0
        
        col[mask] = stat
        X_imputed[:, j] = col
    
    return X_imputed
