import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    classes = np.unique(y_train)
    n_classes = len(classes)

    class_counts = np.array([np.sum(y_train == c) for c in classes])
    class_priors = class_counts / n_train
    log_class_priors = np.log(class_priors)

    theta = np.zeros((n_classes, d))
    for c_idx, cls in enumerate(classes):
        mask = y_train == cls
        X_class = X_train[mask]
        n_y = X_class.shape[0]
        ones_count = np.sum(X_class, axis=0)
        theta[c_idx] = (ones_count + 1) / (n_y + 2)

    log_theta = np.log(theta)
    log_1_minus_theta = np.log(1 - theta)

    log_likelihoods = np.zeros((n_test, n_classes))
    
    for c_idx in range(n_classes):
        log_likelihoods[:, c_idx] = np.sum(
            X_test * log_theta[c_idx] + (1 - X_test) * log_1_minus_theta[c_idx],
            axis=1
        )

    log_posteriors = log_likelihoods + log_class_priors
    return log_posteriors