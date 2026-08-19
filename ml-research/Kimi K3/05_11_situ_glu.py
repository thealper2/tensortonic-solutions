import torch

def situ_glu(input_tensor, gate_projection, up_projection, gate_cap=4.0, up_cap=25.0):
    """
    Returns: the bounded element-wise gated activation.
    """
    gate = input_tensor @ gate_projection.T
    up = input_tensor @ up_projection.T
    gate_capped = gate_cap * torch.tanh(gate / gate_cap)
    up_capped = up_cap * torch.tanh(up / up_cap)
    gate_branch = gate_capped * torch.sigmoid(gate)
    output = gate_branch * up_capped
    return output