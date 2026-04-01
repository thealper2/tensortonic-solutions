import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w = np.array(w)
    v = np.array(v)
    grad = np.array(grad)

    new_v = momentum * v + lr * grad
    new_w = w - new_v
    return new_w, new_v
