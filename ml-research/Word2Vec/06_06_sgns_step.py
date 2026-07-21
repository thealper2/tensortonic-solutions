import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()
    
    v_c = W_in[center_id].clone()
    
    u_o = W_out[pos_id]
    score_o = torch.dot(v_c, u_o)
    grad_o = torch.sigmoid(score_o) - 1.0
    
    W_out_updated[pos_id] = u_o - lr * grad_o * v_c
    
    grad_v = grad_o * u_o
    neg_ids_unique = torch.unique(neg_ids)
    
    for neg_id in neg_ids_unique:
        mask = (neg_ids == neg_id)
        count = mask.sum().item()
        
        u_n = W_out[neg_id]
        score_n = torch.dot(v_c, u_n)
        grad_n = torch.sigmoid(score_n)
        
        W_out_updated[neg_id] = u_n - lr * grad_n * v_c * count
        
        grad_v += grad_n * u_n * count
    
    W_in_updated[center_id] = v_c - lr * grad_v
    
    return W_in_updated, W_out_updated
