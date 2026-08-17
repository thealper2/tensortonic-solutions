def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    result = list(values)
    n = len(result)
    
    i = 0
    while i < n:
        if result[i] is None:
            left_idx = i - 1
            while left_idx >= 0 and result[left_idx] is None:
                left_idx -= 1
            
            right_idx = i + 1
            while right_idx < n and result[right_idx] is None:
                right_idx += 1
            
            left_val = result[left_idx]
            right_val = result[right_idx]
            
            span = right_idx - left_idx
            for j in range(left_idx + 1, right_idx):
                fraction = (j - left_idx) / span
                result[j] = left_val + fraction * (right_val - left_val)
            
            i = right_idx
        else:
            i += 1
    
    return result