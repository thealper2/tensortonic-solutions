import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    N = len(docs)
    if N == 0:
        return np.array([])

    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths)

    unique_query_terms = set(query_tokens)

    doc_freq = {}
    for term in unique_query_terms:
        df = 0
        for doc in docs:
            if term in doc:
                df += 1
        doc_freq[term] = df

    term_freqs = []
    for doc in docs:
        term_freqs.append(Counter(doc))

    idf = {}
    for term in unique_query_terms:
        df = doc_freq[term]
        idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    scores = np.zeros(N, dtype=float)

    for i, (doc, tf_counter) in enumerate(zip(docs, term_freqs)):
        doc_len = doc_lengths[i]
        score = 0.0
        for term in unique_query_terms:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            tf_saturated = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            score += idf[term] * tf_saturated

        scores[i] = score

    return scores