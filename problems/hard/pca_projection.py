import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X)
    n, d = X.shape
    
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    eigen_vals, eigen_vecs = np.linalg.eigh(cov)
    
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]
    
    W = eigen_vecs[:, :k]
    
    X_proj = X_centered @ W
    
    return X_proj.tolist()