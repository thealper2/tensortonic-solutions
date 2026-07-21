import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    dtype = x.dtype
    device = x.device
    
    def to_tensor(t, shape=None):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=dtype)
        return t.to(device)
    
    bn1_gamma = to_tensor(bn1_gamma)
    bn1_beta = to_tensor(bn1_beta)
    bn1_mean = to_tensor(bn1_mean)
    bn1_var = to_tensor(bn1_var)
    conv1_weight = to_tensor(conv1_weight)
    bn2_gamma = to_tensor(bn2_gamma)
    bn2_beta = to_tensor(bn2_beta)
    bn2_mean = to_tensor(bn2_mean)
    bn2_var = to_tensor(bn2_var)
    conv2_weight = to_tensor(conv2_weight)
    
    x_norm1 = (x - bn1_mean.view(1, -1, 1, 1)) / torch.sqrt(bn1_var.view(1, -1, 1, 1) + eps)
    x_norm1 = bn1_gamma.view(1, -1, 1, 1) * x_norm1 + bn1_beta.view(1, -1, 1, 1)
    x_relu1 = F.relu(x_norm1)
    h = F.conv2d(x_relu1, conv1_weight, bias=None, stride=1, padding=0)
    
    x_norm2 = (h - bn2_mean.view(1, -1, 1, 1)) / torch.sqrt(bn2_var.view(1, -1, 1, 1) + eps)
    x_norm2 = bn2_gamma.view(1, -1, 1, 1) * x_norm2 + bn2_beta.view(1, -1, 1, 1)
    x_relu2 = F.relu(x_norm2)
    out = F.conv2d(x_relu2, conv2_weight, bias=None, stride=1, padding=1)
    
    return out
