import torch

def _read_depth_sources(sources, pseudo_query, eps):
    normalized = sources / torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    logits = (normalized * pseudo_query).sum(dim=-1)
    weights = torch.softmax(logits, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights

def block_attention_residual(embedding, previous_outputs, pseudo_query, block_size, eps=1e-6):
    """
    Returns: retrieved values, depth weights, and block-level sources.
    """
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding, dtype=torch.float32)
    
    if not isinstance(pseudo_query, torch.Tensor):
        pseudo_query = torch.tensor(pseudo_query, dtype=torch.float32)
    
    block_sources = []
    block_sources.append(embedding)
    
    if previous_outputs is not None and len(previous_outputs) > 0:
        prev_tensors = []
        for out in previous_outputs:
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, dtype=torch.float32)
            prev_tensors.append(out)
        
        prev_stack = torch.stack(prev_tensors, dim=0)
        n = prev_stack.shape[0]
        complete_layers = n - (n % block_size)
        
        if complete_layers > 0:
            complete = prev_stack[:complete_layers]
            for i in range(0, complete_layers, block_size):
                block_sum = complete[i:i+block_size].sum(dim=0)
                block_sources.append(block_sum)
        
        remainder = prev_stack[complete_layers:]
        if remainder.shape[0] > 0:
            partial_sum = remainder.sum(dim=0)
            block_sources.append(partial_sum)
    
    sources = torch.stack(block_sources, dim=0)
    retrieved, weights = _read_depth_sources(sources, pseudo_query, eps)
    return retrieved, weights, sources