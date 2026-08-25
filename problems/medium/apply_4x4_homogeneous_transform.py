import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T)
    points = np.asarray(points)

    if points.ndim == 1:
        points_h = np.append(points, 1.0)
        transformed_h = T @ points_h
        return transformed_h[:3]

    else:
        N = points.shape[0]
        ones = np.ones((N, 1))
        points_h = np.hstack([points, ones])
        trasnformed_h = points_h @ T.T
        return trasnformed_h[:, :3]