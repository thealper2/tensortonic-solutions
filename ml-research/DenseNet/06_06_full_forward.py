import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-3x3Conv (padding 1, no bias) producing growth_rate channels.
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
    
    out = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)
    
    return out


def dense_block(x, layers, eps):
    """
    Returns torch.Tensor: concat of x and every composite-layer output (channels grow by growth_rate per layer).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    feats = [x]
    
    for layer in layers:
        cat_input = torch.cat(feats, dim=1)
        out = composite_layer(
            cat_input,
            layer['bn_gamma'],
            layer['bn_beta'],
            layer['bn_mean'],
            layer['bn_var'],
            layer['conv_weight'],
            eps
        )
        feats.append(out)
    
    return torch.cat(feats, dim=1)


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-1x1Conv then 2x2 average pool with stride 2.
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


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    dtype = x.dtype
    device = x.device
    
    def to_tensor(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=dtype)
        return t.to(device)
    
    stem_conv = to_tensor(weights['stem_conv'])
    x = F.conv2d(x, stem_conv, bias=None, stride=1, padding=1)
    
    blocks = weights['blocks']
    transitions = weights['transitions']
    
    for i, block_layers in enumerate(blocks):
        x = dense_block(x, block_layers, eps)
        
        if i < len(blocks) - 1:
            trans = transitions[i]
            x = transition_layer(
                x,
                trans['bn_gamma'],
                trans['bn_beta'],
                trans['bn_mean'],
                trans['bn_var'],
                trans['conv_weight'],
                eps
            )
    
    final_bn_gamma = to_tensor(weights['final_bn_gamma'])
    final_bn_beta = to_tensor(weights['final_bn_beta'])
    final_bn_mean = to_tensor(weights['final_bn_mean'])
    final_bn_var = to_tensor(weights['final_bn_var'])
    
    x_norm = (x - final_bn_mean.view(1, -1, 1, 1)) / torch.sqrt(final_bn_var.view(1, -1, 1, 1) + eps)
    x_norm = final_bn_gamma.view(1, -1, 1, 1) * x_norm + final_bn_beta.view(1, -1, 1, 1)
    x_relu = F.relu(x_norm)
    
    pooled = x_relu.mean(dim=(2, 3))
    
    fc_weight = to_tensor(weights['fc_weight'])
    fc_bias = to_tensor(weights['fc_bias'])
    logits = pooled @ fc_weight.T + fc_bias
    
    return logits
