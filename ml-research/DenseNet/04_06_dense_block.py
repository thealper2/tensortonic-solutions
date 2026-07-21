import torch

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    dtype = x.dtype
    device = x.device
    
    def to_tensor(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=dtype)
        return t.to(device)
    
    feats = [x]
    current_channels = x.shape[1]
    
    for layer in layers:
        bn_gamma = to_tensor(layer['bn_gamma'])
        bn_beta = to_tensor(layer['bn_beta'])
        bn_mean = to_tensor(layer['bn_mean'])
        bn_var = to_tensor(layer['bn_var'])
        conv_weight = to_tensor(layer['conv_weight'])
        
        cat_input = torch.cat(feats, dim=1)
        
        x_norm = (cat_input - bn_mean.view(1, -1, 1, 1)) / torch.sqrt(bn_var.view(1, -1, 1, 1) + eps)
        x_norm = bn_gamma.view(1, -1, 1, 1) * x_norm + bn_beta.view(1, -1, 1, 1)
        x_relu = F.relu(x_norm)
        out = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)
        
        feats.append(out)
    
    return torch.cat(feats, dim=1)
