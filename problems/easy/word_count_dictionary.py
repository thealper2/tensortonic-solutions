def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    freq = {}
    for sent in sentences:
        for token in sent:
            freq[token] = freq.get(token, 0) + 1
            
    return freq
