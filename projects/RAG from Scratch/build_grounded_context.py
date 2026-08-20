def build_context(retrieved_chunks, max_words):
    """
    Returns: labeled context containing the longest ranked prefix that fits
    """
    included = []
    total_words = 0

    for chunk in retrieved_chunks:
        text = chunk["text"]
        word_count = len(text.split())

        if total_words + word_count > max_words:
            break
            
        included.append(chunk)
        total_words += word_count

    if not included:
        return ""

    blocks = []
    for chunk in included:
        block = f"[{chunk['id']}]\n{chunk['text']}"
        blocks.append(block)

    return "\n\n".join(blocks)