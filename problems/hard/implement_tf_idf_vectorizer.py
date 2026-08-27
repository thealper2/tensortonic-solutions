import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    tokenized_docs = []
    for doc in documents:
        tokens = doc.lower().split()
        tokenized_docs.append(tokens)

    all_terms = set()
    for tokens in tokenized_docs:
        all_terms.update(tokens)

    vocabulary = sorted(all_terms)
    V = len(vocabulary)
    N = len(documents)

    term_to_idx = {term: i for i, term in enumerate(vocabulary)}

    doc_freq = Counter()
    for tokens in tokenized_docs:
        unique_terms = set(tokens)
        doc_freq.update(unique_terms)

    tfidf_matrix = np.zeros((N, V), dtype=float)

    for i, tokens in enumerate(tokenized_docs):
        doc_len = len(tokens)
        tf_counts = Counter(tokens)
        for term, count in tf_counts.items():
            tf = count / doc_len
            df = doc_freq[term]
            idf = math.log(N / df)
            tfidf_matrix[i, term_to_idx[term]] = tf * idf

    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary,
    }