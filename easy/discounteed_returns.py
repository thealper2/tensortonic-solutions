def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    T = len(rewards)
    G = [0.0] * T
    if T == 0:
        return G

    G[-1] = float(rewards[-1])
    for t in range(T - 2, -1, -1):
        G[t] = float(rewards[t]) + gamma * G[t + 1]

    return G
