import numpy as np

def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    U = np.array(U)
    V = np.array(V)
    e = r - np.dot(U, V)
    U_i = U + lr * (e * V - reg * U)
    V_i = V + lr * (e * U - reg * V)
    return U_i.tolist(), V_i.tolist()
