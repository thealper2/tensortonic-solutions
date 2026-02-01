import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states
    
    for episode in episodes:
        G = 0.0
        visited = set()
        n = len(episode)
        for t in range(n - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1

    V = [0.0] * n_states
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]

    V = np.array(V)
    return V
