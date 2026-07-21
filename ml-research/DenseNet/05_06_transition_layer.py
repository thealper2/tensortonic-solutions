import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    dtype = x.dtype
    device = x.device
    
    def to_tensor(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=dtype)
        return t.to(device)
    
    bn_gamma = to_tensor(bn_gamma)
    bn_beta = to_tensor(bn_beta)
    bn_mean = to_tensor(bn_mean)
    bn_var = to_tensor(bn_var)
    conv_weight = to_tensor(conv_weight)
    
    x_norm = (x - bn_mean.view(1, -1, 1, 1)) / torch.sqrt(bn_var.view(1, -1, 1, 1) + eps)
    x_norm = bn_gamma.view(1, -1, 1, 1) * x_norm + bn_beta.view(1, -1, 1, 1)
    x_relu = F.relu(x_norm)
    
    y = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=0)
    out = F.avg_pool2d(y, kernel_size=2, stride=2)
    
    return out
