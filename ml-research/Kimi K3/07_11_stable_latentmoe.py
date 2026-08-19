import torch

def _situ_glu(x, Wg, Wu, Wd, gate_cap, up_cap):
    """SiTU-GLU expert: softcapped gated linear unit, then down-projection."""
    g = x @ Wg.T
    u = x @ Wu.T
    softcap_g = gate_cap * torch.tanh(g / gate_cap)
    softcap_u = up_cap * torch.tanh(u / up_cap)
    sigmoid_g = 0.5 * (1.0 + torch.tanh(0.5 * g))
    return (softcap_g * sigmoid_g * softcap_u) @ Wd.T

def stable_latent_moe(tokens, latent_down_projection, latent_up_projection, router_projection, current_bias, routed_gate_weights, routed_up_weights, routed_down_weights, shared_gate_weights, shared_up_weights, shared_down_weights, selected_count, eps=1e-6, gate_cap=4.0, up_cap=25.0):
    """
    Returns: final output, routes, mixture weights, and the latent routed aggregate.
    """
    x = torch.as_tensor(tokens)
    out_device = x.device
    out_dtype = x.dtype if x.is_floating_point() else torch.float64
    dtype = out_dtype
    x = x.to(dtype)

    def cvt(a): return torch.as_tensor(a, dtype=dtype, device=out_device)
    Wd, Wu = cvt(latent_down_projection), cvt(latent_up_projection)
    Wr, bias = cvt(router_projection), cvt(current_bias)
    rg, ru, rd = cvt(routed_gate_weights), cvt(routed_up_weights), cvt(routed_down_weights)
    sg, su, sd = cvt(shared_gate_weights), cvt(shared_up_weights), cvt(shared_down_weights)

    T, d = x.shape
    k = selected_count
    r = torch.sigmoid(x @ Wr.T)
    sel = torch.topk(r + bias, k, dim=1).indices
    raw = r.gather(1, sel)
    p = raw / raw.sum(dim=1, keepdim=True)
    z = x @ Wd.T
    u_agg = torch.zeros_like(z)
    for ti in range(T):
        for slot in range(k):
            ei = int(sel[ti, slot])
            u_agg[ti] += p[ti, slot] * _situ_glu(z[ti], rg[ei], ru[ei], rd[ei], gate_cap, up_cap)

    rms = torch.sqrt(u_agg.square().mean(dim=-1, keepdim=True) + eps)
    up = (u_agg / rms) @ Wu.T
    shared = _situ_glu(x, sg[0], su[0], sd[0], gate_cap, up_cap) + _situ_glu(x, sg[1], su[1], sd[1], gate_cap, up_cap)

    y = shared + up
    return (y.to(dtype=out_dtype, device=out_device),
            sel.to(device=out_device),
            p.to(dtype=out_dtype, device=out_device),
            u_agg.to(dtype=out_dtype, device=out_device))