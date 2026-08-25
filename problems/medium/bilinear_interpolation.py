def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    
    if new_h == 1:
        y_scale = 0.0
    else:
        y_scale = (H - 1) / (new_h - 1)
    
    if new_w == 1:
        x_scale = 0.0
    else:
        x_scale = (W - 1) / (new_w - 1)
    
    output = []
    
    for i in range(new_h):
        row = []
        y = i * y_scale
        y0 = int(y)
        y1 = min(y0 + 1, H - 1)
        dy = y - y0
        
        for j in range(new_w):
            x = j * x_scale
            x0 = int(x)
            x1 = min(x0 + 1, W - 1)
            dx = x - x0
            
            v0 = image[y0][x0] * (1 - dx) + image[y0][x1] * dx
            v1 = image[y1][x0] * (1 - dx) + image[y1][x1] * dx
            val = v0 * (1 - dy) + v1 * dy
            
            row.append(val)
        output.append(row)
    
    return output