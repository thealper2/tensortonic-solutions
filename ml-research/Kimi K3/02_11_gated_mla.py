import math
import torch
import torch.nn.functional as F

def gated_mla(hidden_states, query_projection, latent_down_projection, key_up_projection, value_up_projection, output_gate_projection, output_projection, num_heads, causal=True):
    """
    Returns: gated attention outputs and the latent key-value cache.
    """
    B, S, D = hidden_states.shape
    Dh = D // num_heads
    Q = hidden_states @ query_projection.T
    C = hidden_states @ latent_down_projection.T
    K = C @ key_up_projection.T
    V = C @ value_up_projection.T
    Q = Q.reshape(B, S, num_heads, Dh).transpose(1, 2)
    K = K.reshape(B, S, num_heads, Dh).transpose(1, 2)
    V = V.reshape(B, S, num_heads, Dh).transpose(1, 2)
    scale = Dh ** 0.5
    scores = (Q @ K.transpose(-2, -1)) / scale

    if causal:
        mask = torch.triu(torch.ones(S, S, device=hidden_states.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)
    context = attn_weights @ V
    context = context.transpose(1, 2).reshape(B, S, D)
    gate = torch.sigmoid(hidden_states @ output_gate_projection.T)
    gated = gate * context
    output = gated @ output_projection.T
    return output, C