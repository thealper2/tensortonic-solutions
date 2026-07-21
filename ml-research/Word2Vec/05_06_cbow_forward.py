import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    h = W_in[context_ids].mean(dim=0)
    logits = W_out @ h
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], dtype=torch.long))
    return loss
