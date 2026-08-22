def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    vocab = set(tokens)
    V = len(vocab)

    counts = {}
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1

    probs = {}
    for w1 in vocab:
        total_count = sum(counts.get((w1, w2), 0) for w2 in vocab)
        denominator = total_count + V

        for w2 in vocab:
            count = counts.get((w1, w2), 0)
            probs[(w1, w2)] = (count + 1) / denominator

    return counts, probs