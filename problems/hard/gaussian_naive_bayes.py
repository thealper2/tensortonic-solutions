import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    n_train, d = X_train.shape
    n_test = X_test.shape[0]
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    eps = 1e-9
    
    class_counts = np.array([np.sum(y_train == c) for c in classes])
    class_priors = class_counts / n_train
    log_class_priors = np.log(class_priors)
    
    means = np.zeros((n_classes, d))
    variances = np.zeros((n_classes, d))
    
    for c_idx, cls in enumerate(classes):
        mask = y_train == cls
        X_class = X_train[mask]
        n_c = X_class.shape[0]
    
        means[c_idx] = np.mean(X_class, axis=0)
        variances[c_idx]= np.var(X_class, axis=0, ddof=0)
        variances[c_idx] += eps
    
    log_posteriors = np.zeros((n_test, n_classes))
    
    for c_idx in range(n_classes):
        mu = means[c_idx]
        var = variances[c_idx]
    
        log_likelihood = -0.5 * np.log(2 * np.pi * var) - (X_test - mu) ** 2 / (2 * var)
    
        log_posteriors[:, c_idx] = np.sum(log_likelihood, axis=1) + log_class_priors[c_idx]
    
    predictions = classes[np.argmax(log_posteriors, axis=1)]
    return predictions.tolist()