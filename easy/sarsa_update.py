def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    new_q = [row[:] for row in q_table]
    td_error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    new_q[state][action] += alpha * td_error
    return new_q
