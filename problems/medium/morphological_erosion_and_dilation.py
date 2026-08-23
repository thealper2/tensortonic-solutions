def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    kH = len(kernel)
    kW = len(kernel[0]) if kH > 0 else 0
    
    pad_h = kH // 2
    pad_w = kW // 2
    
    padded = [[0] * (W + 2 * pad_w) for _ in range(H + 2 * pad_h)]
    for i in range(H):
        for j in range(W):
            padded[i + pad_h][j + pad_w] = image[i][j]
    
    output = [[0] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if operation == "erode":
                match = True
                for ki in range(kH):
                    for kj in range(kW):
                        if kernel[ki][kj] == 1:
                            if padded[i + ki][j + kj] != 1:
                                match = False
                                break
                                
                    if not match:
                        break
                        
                output[i][j] = 1 if match else 0
            
            elif operation == "dilate":
                match = False
                for ki in range(kH):
                    for kj in range(kW):
                        if kernel[ki][kj] == 1:
                            if padded[i + ki][j + kj] == 1:
                                match = True
                                break
                                
                    if match:
                        break
                        
                output[i][j] = 1 if match else 0
    
    return output