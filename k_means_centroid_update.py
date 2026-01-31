def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    n = len(points)
    if n == 0:
        return [[0.0] * len(points[0]) if points else [0.0]] * k
    
    d = len(points[0])
    centroids = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    
    for i in range(n):
        cluster = assignments[i]
        counts[cluster] += 1
        for j in range(d):
            centroids[cluster][j] += points[i][j]
    
    for cluster in range(k):
        if counts[cluster] > 0:
            for j in range(d):
                centroids[cluster][j] /= counts[cluster]
    
    return centroids
