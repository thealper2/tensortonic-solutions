import torch

def kda_context_parallel(transition_matrices, local_contributions, initial_state):
    """
    Returns: incoming states, outgoing states, and the final global state.
    """
    L = transition_matrices.shape[0]
    W = transition_matrices.shape[1]
    V = local_contributions.shape[2]

    device = initial_state.device
    dtype = initial_state.dtype

    prefix_transition = torch.eye(W, dtype=dtype, device=device)
    prefix_contribution = torch.zeros(W, V, dtype=dtype, device=device)

    incoming_states = torch.zeros(L, W, V, dtype=dtype, device=device)
    outgoing_states = torch.zeros(L, W, V, dtype=dtype, device=device)

    for i in range(L):
        incoming = prefix_transition @ initial_state + prefix_contribution
        incoming_states[i] = incoming

        M_i = transition_matrices[i]
        U_i = local_contributions[i]
        outgoing = M_i @ incoming + U_i
        outgoing_states[i] = outgoing

        prefix_transition = M_i @ prefix_transition
        prefix_contribution = M_i @ prefix_contribution + U_i

    final_state = outgoing_states[-1]
    return incoming_states, outgoing_states, final_state