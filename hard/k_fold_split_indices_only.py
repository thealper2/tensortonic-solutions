import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    if rng:
        arr = rng.permutation(N)
    else:
        arr = np.arange(N)
        np.random.shuffle(arr)

    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[:N % k] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        start = current
        end = current + fold_size

        val_idx = arr[start:end]
        train_idx = np.concatenate([arr[:start], arr[end:]])

        folds.append((train_idx, val_idx))

        current = end

    return folds
