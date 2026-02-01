import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    n = len(states)
    advantages = [0.0] * n
    returns = [0.0] * n

    G = 0
    for t in range(n - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(n):
        advantages[t] = returns[t] - V[states[t]]

    return np.array(advantages)
