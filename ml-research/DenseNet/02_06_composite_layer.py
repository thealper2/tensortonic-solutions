import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    bn_gamma = torch.as_tensor(bn_gamma, dtype=x.dtype)
    bn_beta = torch.as_tensor(bn_beta, dtype=x.dtype)
    bn_mean = torch.as_tensor(bn_mean, dtype=x.dtype)
    bn_var = torch.as_tensor(bn_var, dtype=x.dtype)
    conv_weight = torch.as_tensor(conv_weight, dtype=x.dtype)

    gamma = bn_gamma.reshape(1, -1, 1, 1)
    beta = bn_beta.reshape(1, -1, 1, 1)
    mean = bn_mean.reshape(1, -1, 1, 1)
    var = bn_var.reshape(1, -1, 1, 1)

    y = gamma * (x - mean) / torch.sqrt(var + eps) + beta
    y = F.relu(y)
    y = F.conv2d(y, conv_weight, bias=None, stride=1, padding=1)

    return y
