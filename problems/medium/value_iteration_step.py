import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    values = np.array(values)
    new_values = []

    for s in range(num_states):
        num_actions = len(rewards[s])
        q_values = np.zeros(num_actions)

        for a in range(num_actions):
            trans = np.array(transitions[s][a])
            q_values[a] = rewards[s][a] + gamma * np.sum(trans * values)

        new_values.append(np.max(q_values))
    
    return new_values