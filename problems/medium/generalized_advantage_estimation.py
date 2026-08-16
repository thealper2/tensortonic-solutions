import numpy as np

def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    T = len(rewards)
    advantages = [0.0] * T
    
    last_adv = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * last_adv
        last_adv = advantages[t]
    
    return advantages