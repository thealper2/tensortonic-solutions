import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points)
    rotated = []

    if points.ndim == 1:
        points = points.reshape(1, 3)

    for point in points:
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        rotated_point = R_z @ point
        rotated.append(rotated_point.tolist())

    rotated = np.asarray(rotated)
    if rotated.shape[0] == 1 and rotated.shape[1] == 3:
        rotated = rotated.reshape(3)
    
    return rotated