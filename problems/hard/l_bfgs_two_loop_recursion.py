import numpy as np

def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    grad = np.array(grad)
    s_list = [np.array(s) for s in s_list]
    y_list = [np.array(y) for y in y_list]

    n = len(grad)
    m = len(s_list)

    rho = []
    for i in range(m):
        rho.append(1.0 / np.dot(y_list[i], s_list[i]))

    q = grad.copy()
    alpha = np.zeros(m)

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * np.dot(s_list[i], q)
        q = q - alpha[i] * y_list[i]

    last_s = s_list[-1]
    last_y = y_list[-1]
    gamma = np.dot(last_s, last_y) / np.dot(last_y, last_y)
    r = gamma * q

    for i in range(m):
        beta = rho[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha[i] - beta)

    return (-r).tolist()