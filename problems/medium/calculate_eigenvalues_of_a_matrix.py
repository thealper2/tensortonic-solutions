import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    all_equal = all(np.shape(item) == np.shape(matrix[0]) for item in matrix)
    if not all_equal:
        return None
        
    matrix = np.array(matrix)
    
    if matrix.ndim == 2:
        if matrix.shape[0] != matrix.shape[1]:
            return None
    elif matrix.ndim == 3:
        n, rows, cols = matrix.shape
        if rows != cols:
            return None
    else:
        return None
    
    return np.linalg.eigvals(matrix)