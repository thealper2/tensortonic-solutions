def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n = len(values)
    freq = {}
    for val in values:
        freq[val] = freq.get(val, 0) + 1

    encoded_values = [freq[val] / n for val in values]
    return encoded_values
