def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    d = {k: i for i, k in enumerate(ordering)}
    return [d[value] for value in values]
