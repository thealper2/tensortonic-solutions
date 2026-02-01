import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("")

    trace = 0
    n = A.shape[0]
    for i in range(n):
        trace += A[i, i]
    
    return trace
