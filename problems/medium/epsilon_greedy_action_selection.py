import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    q_values = np.asarray(q_values)
    n_actions = q_values.shape[0]

    if rng is None:
        rand = np.random.random()
    else:
        rand = rng.random()

    if rand < epsilon:
        if rng is None:
            return int(np.random.randint(n_actions))
        else:
            return int(rng.integers(n_actions))

    return int(np.argmax(q_values))
