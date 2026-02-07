import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    N = X.shape[0]

    unique_labels = np.unique(labels)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    a = np.zeros(N)
    b = np.full(N, np.inf)

    for label in unique_labels:
        in_cluster = labels == label
        out_cluster = ~in_cluster
        cluster_size = np.sum(in_cluster)

        if cluster_size > 1:
            a[in_cluster] = np.sum(dist[in_cluster][:, in_cluster], axis=1) / (cluster_size - 1)
        else:
            a[in_cluster] = 0.0

        for other_label in unique_labels:
            if other_label == label:
                continue

            other_cluster = labels == other_label
            mean_dist = np.mean(dist[in_cluster][:, other_cluster], axis=1)
            b[in_cluster] = np.minimum(b[in_cluster], mean_dist)

    s = (b - a) / np.maximum(a, b)
    return float(np.mean(s))
