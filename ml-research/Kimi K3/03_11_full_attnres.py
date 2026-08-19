import torch

def full_attention_residual(embedding, previous_outputs, pseudo_query, eps=1e-6):
    """
    Returns: retrieved representations and depth-attention weights.
    """
    sources = [embedding.unsqueeze(0)]
    if previous_outputs is not None and len(previous_outputs) > 0:
        for out in previous_outputs:
            sources.append(out.unsqueeze(0))

    sources = torch.cat(sources, dim=0)
    rms = torch.sqrt(sources.square().mean(dim=-1, keepdim=True) + eps)
    keys_norm = sources / rms
    scores = (keys_norm * pseudo_query).sum(dim=-1)
    weights = torch.softmax(scores, dim=0)
    retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
    return retrieved, weights