import torch

def quantile_balancing(router_scores, current_bias, selected_count):
    """
    Returns: selected experts, mixture weights, loads, and the next centered bias.
    """
    m, n = router_scores.shape
    k = selected_count
    target_load = (m * k) // n
    biased = router_scores + current_bias
    sorted_indices = torch.argsort(biased, dim=-1, descending=True, stable=True)
    selected = sorted_indices[:, :k]
    cutoff_indices = sorted_indices[:, k:k+1]
    cutoffs = biased.gather(1, cutoff_indices).squeeze(-1)
    raw_selected = router_scores.gather(1, selected)
    weights = raw_selected / raw_selected.sum(dim=-1, keepdim=True)
    loads = torch.zeros(n, dtype=torch.long, device=router_scores.device)
    for i in range(m):
        for j in range(k):
            loads[selected[i, j]] += 1
    
    margins = router_scores - cutoffs.unsqueeze(-1)
    sorted_margins, _ = torch.sort(margins, dim=0, descending=True)
    threshold_margin = sorted_margins[target_load, :]
    next_bias = -threshold_margin
    next_bias = next_bias - next_bias.mean()
    return selected, weights, loads, next_bias