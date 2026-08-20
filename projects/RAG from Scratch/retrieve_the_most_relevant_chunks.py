import math


def retrieve_top_k(query_embedding, chunks, k):
    """
    Returns: ranked copies of the k most similar chunk records with scores
    """
    query_norm = math.sqrt(sum(q * q for q in query_embedding))
    scored_chunks = []

    for idx, chunk in enumerate(chunks):
        chunk_embedding = chunk["embedding"]
        dot_product = sum(q * c for q, c in zip(query_embedding, chunk_embedding))
        chunk_norm = math.sqrt(sum(c * c for c in chunk_embedding))

        if query_norm == 0 or chunk_norm == 0:
            score = 0.0
        else:
            score = dot_product / (query_norm * chunk_norm)

        new_chunk = dict(chunk)
        new_chunk["score"] = score
        scored_chunks.append((score, idx, new_chunk))

    scored_chunks.sort(key=lambda x: (-x[0], x[1]))
    return [item[2] for item in scored_chunks[:k]]