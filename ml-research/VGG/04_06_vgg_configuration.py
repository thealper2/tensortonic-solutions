import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant.
    """
    variant = variant.lower()
    configs = {
        'vgg11': [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
        'vgg13': [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)],
        'vgg16': [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
        'vgg19': [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)],
    }

    config = []
    for num_convs, num_filters in configs[variant]:
        config.extend([num_filters] * num_convs)
        config.append('M')

    return config
