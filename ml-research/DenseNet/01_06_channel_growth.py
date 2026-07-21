import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    counts = [stem_channels]
    C = stem_channels

    for idx, n in enumerate(block_layers):
        C = C + n * growth_rate
        counts.append(C)

        if idx != len(block_layers) - 1:
            C = int(math.floor(C * compression))
            counts.append(C)

    return torch.tensor(counts, dtype=torch.int64)
