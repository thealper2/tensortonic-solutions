import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q_arr = np.asarray(Q, dtype=float)
    Q_new = Q_arr.copy()

    td_target = r + gamma * np.max(Q_arr[s_next])
    Q_new[s, a] += alpha * (td_target - Q_new[s, a])

    return Q_new
