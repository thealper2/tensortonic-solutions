import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    total = counts.sum().float()
    freq = counts.float() / total
    keep_probs = torch.sqrt(t / freq)
    keep_probs = torch.clamp(keep_probs, max=1.0)
    return keep_probs
