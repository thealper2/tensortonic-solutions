import torch

def kda_recurrence(query, key, value, decay_logits, write_strength, output_gate_logits, output_projection, initial_state, g_min=-5.0, eps=1e-6):
    """
    Returns: sequence outputs and the final recurrent state.
    """
    B, S, H, Dk = query.shape
    Dv = value.shape[-1]
    device = query.device
    dtype = query.dtype
    
    if write_strength.dim() == 4:
        write_strength = write_strength.squeeze(-1)
    
    if decay_logits.dim() == 3:
        decay_logits = decay_logits.unsqueeze(-1).expand(B, S, H, Dk)
    
    state = initial_state.clone()
    outputs = torch.zeros(B, S, output_projection.shape[0], dtype=dtype, device=device)
    
    for t in range(S):
        q_t = query[:, t, :, :]
        k_t = key[:, t, :, :]
        v_t = value[:, t, :, :]
        z_t = decay_logits[:, t, :, :]
        beta_t = write_strength[:, t, :]
        gate_t = output_gate_logits[:, t, :, :]
        alpha = torch.exp(g_min * torch.sigmoid(z_t))
        alpha_exp = alpha.unsqueeze(-1)
        state_decayed = alpha_exp * state
        kT_S = (k_t.unsqueeze(-2) @ state_decayed)
        k_kT_S = k_t.unsqueeze(-1) * kT_S
        beta_exp = beta_t.unsqueeze(-1).unsqueeze(-1)
        state_erased = state_decayed - beta_exp * k_kT_S
        k_vT = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        state = state_erased + beta_exp * k_vT
        qT_S = (q_t.unsqueeze(-2) @ state)
        o_t = qT_S.squeeze(-2)
        rms = torch.sqrt((o_t ** 2).mean(dim=-1, keepdim=True) + eps)
        o_t_norm = o_t / rms
        gate = torch.sigmoid(gate_t)
        o_gated = gate * o_t_norm
        o_flat = o_gated.reshape(B, H * Dv)
        outputs[:, t, :] = o_flat @ output_projection.T
    
    return outputs, state