import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    output = [[0] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):
            dy = i - cy
            dx = j - cx

            src_y = cy + dy * cos_theta + dx * sin_theta
            src_x = cx - dy * sin_theta + dx * cos_theta

            sy = int(round(src_y))
            sx = int(round(src_x))

            if 0 <= sy < H and 0 <= sx < W:
                output[i][j] = image[sy][sx]
            else:
                output[i][j] = 0

    return output