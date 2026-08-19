import torch

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def mxfp4_expert_linear(latent_tokens, packed_weights, scale_bytes, selected_experts, mixture_weights, shared_output):
    """
    Returns: combined routed and shared expert output.
    """
    T, I = latent_tokens.shape
    _, O, G, _ = packed_weights.shape
    K = selected_experts.shape[1]
    
    device = latent_tokens.device
    dtype = latent_tokens.dtype
    
    e2m1 = torch.tensor(_E2M1_VALUES, dtype=dtype, device=device)
    
    routed_output = torch.zeros(T, O, dtype=dtype, device=device)
    
    for t in range(T):
        for k in range(K):
            expert_idx = int(selected_experts[t, k].item())
            weight = mixture_weights[t, k].item()
            
            if weight == 0:
                continue
            
            packed = packed_weights[expert_idx]
            scales = scale_bytes[expert_idx]
            W_expert = torch.zeros(O, I, dtype=dtype, device=device)
            
            for o in range(O):
                for g in range(G):
                    bytes_vals = packed[o, g]
                    scale = 2.0 ** (scales[o, g].float() - 127.0)
                    
                    vals = torch.zeros(32, dtype=dtype, device=device)
                    for b in range(16):
                        byte_val = bytes_vals[b].item()
                        low_nibble = byte_val & 0x0F
                        high_nibble = (byte_val >> 4) & 0x0F
                        vals[2*b] = e2m1[low_nibble]
                        vals[2*b + 1] = e2m1[high_nibble]
                    
                    vals = vals * scale
                    
                    start_idx = g * 32
                    W_expert[o, start_idx:start_idx + 32] = vals
            
            routed_output[t] += weight * (W_expert @ latent_tokens[t])
    
    combined = shared_output + routed_output
    return combined