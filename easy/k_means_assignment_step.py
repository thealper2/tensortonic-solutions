import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)
    centroids = np.array(centroids)
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2).astype("float32")
    return np.argmin(distances, axis=1).tolist()
