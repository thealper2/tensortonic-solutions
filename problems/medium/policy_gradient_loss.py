import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    log_probs = np.array(log_probs, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards)
    discounted_sum = 0.0
    for t in reversed(range(len(rewards))):
        discounted_sum = rewards[t] + gamma * discounted_sum
        returns[t] = discounted_sum

    baseline = np.mean(returns)
    advantages = returns - baseline
    loss = -np.mean(log_probs * advantages)
    return float(loss)