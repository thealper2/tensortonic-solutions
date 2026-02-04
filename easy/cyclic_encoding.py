import math

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    encoded = []
    for v in values:
        angle = 2 * math.pi * v / period
        encoded.append([math.sin(angle), math.cos(angle)])

    return encoded
