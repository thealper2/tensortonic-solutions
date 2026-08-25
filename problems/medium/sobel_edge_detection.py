import math

def sobel_edges(image: list) -> list:
    """
    Returns the zero-padded Sobel gradient magnitude at every pixel.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    padded = [[0] * (W + 2) for _ in range(H + 2)]
    for i in range(H):
        for j in range(W):
            padded[i + 1][j + 1] = image[i][j]
    
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    output = [[0.0] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            Gx = 0.0
            Gy = 0.0
            for a in range(3):
                for b in range(3):
                    val = padded[i + a][j + b]
                    Gx += Kx[a][b] * val
                    Gy += Ky[a][b] * val
                    
            magnitude = math.sqrt(Gx * Gx + Gy * Gy)
            output[i][j] = magnitude
    
    return output