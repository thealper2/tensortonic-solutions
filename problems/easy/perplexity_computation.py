import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    n = len(actual_tokens)
    if n == 0:
        return float("inf")

    log_sum = 0.0
    for probs, token in zip(prob_distributions, actual_tokens):
        p = probs[token]
        if p <= 0.0:
            return float("inf")

        log_sum += math.log(p)

    perplexity = math.exp(-log_sum / n)
    return perplexity
