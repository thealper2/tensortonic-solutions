import torch

def per_head_muon(parameter, gradient, previous_momentum, num_heads, momentum_coefficient, learning_rate):
    """
    Returns: updated parameter, momentum, and per-head orthogonalized update.
    """
    rows, cols = parameter.shape
    head_rows = rows // num_heads
    momentum = momentum_coefficient * previous_momentum + gradient
    head_updates = []
    
    for h in range(num_heads):
        start_row = h * head_rows
        end_row = (h + 1) * head_rows
        head_momentum = momentum[start_row:end_row, :]
        U, S, Vh = torch.linalg.svd(head_momentum, full_matrices=False)
        O_h = U @ Vh
        head_updates.append(O_h)
    
    orthogonal_update = torch.cat(head_updates, dim=0)
    updated_parameter = parameter - learning_rate * orthogonal_update
    return updated_parameter, momentum, orthogonal_update