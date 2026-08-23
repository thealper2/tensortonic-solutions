def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    total_pixels = H * W
    
    hist = [0] * 256
    for i in range(H):
        for j in range(W):
            hist[image[i][j]] += 1
    
    cdf = [0] * 256
    running_sum = 0
    for i in range(256):
        running_sum += hist[i]
        cdf[i] = running_sum
    
    cdf_min = None
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break
    
    if cdf_min == total_pixels:
        return [[0] * W for _ in range(H)]
    
    mapping = [0] * 256
    for v in range(256):
        if cdf[v] > 0:
            mapping[v] = round((cdf[v] - cdf_min) / (total_pixels - cdf_min) * 255)
    
    output = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            output[i][j] = mapping[image[i][j]]
    
    return output