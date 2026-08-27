import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    if not candidate:
        return 0.0

    c = len(candidate)
    r = len(reference)

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = []
        for i in range(len(candidate) - n + 1):
            cand_ngrams.append(tuple(candidate[i:i+n]))

        ref_ngrams = []
        for i in range(len(reference) - n + 1):
            ref_ngrams.append(tuple(reference[i:i+n]))

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_sum = 0
        for ngram, count in cand_counts.items():
            clipped_sum += min(count, ref_counts.get(ngram, 0))

        total_cand = len(cand_ngrams)

        if total_cand == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_sum / total_cand)

    if any(p == 0.0 for p in precisions):
        return 0.0

    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    log_sum = sum(math.log(p) for p in precisions)
    bleu = bp * math.exp(log_sum / max_n)

    return bleu