import numpy as np

def update_cell_state(
    c_prev: np.ndarray, 
    f_t: np.ndarray,
    i_t: np.ndarray, 
    c_tilde: np.ndarray
) -> np.ndarray:
    """Update cell state: C_t = f_t * C_prev + i_t * c_tilde"""
    c_next = f_t * c_prev + i_t * c_tilde
    return c_next
