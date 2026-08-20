def chunk_document(text, chunk_size, overlap):
    """
    Returns: normalized overlapping text chunks
    """
    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    if len(words) <= chunk_size:
        return [' '.join(words)]

    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        chunks.append(' '.join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks