import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)
    
    classes, counts = np.unique(y, return_counts=True)
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        class_indices = np.where(y == cls)[0]
        
        if rng is not None:
            shuffled = class_indices.copy()
            rng.shuffle(shuffled)
        else:
            shuffled = class_indices.copy()
            np.random.shuffle(shuffled)
        
        n_class = len(class_indices)
        n_test = int(round(n_class * test_size))
        
        if n_test == 0 and n_class > 1:
            n_test = 1
        
        if n_test == n_class and n_class > 1:
            n_test = n_class - 1
        
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])
    
    train_indices = np.sort(np.array(train_indices))
    test_indices = np.sort(np.array(test_indices))
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test