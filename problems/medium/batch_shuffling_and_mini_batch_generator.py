import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """Yield seeded mini-batches of matching features and labels."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)
    
    if isinstance(drop_last, str):
        drop_last = drop_last.lower() in ('true', '1', 't')
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if drop_last and end - start < batch_size:
            break
            
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]