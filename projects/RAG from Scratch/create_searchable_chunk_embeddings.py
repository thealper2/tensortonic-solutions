import math

def create_chunk_embeddings(chunks, token_embeddings, attention_masks):
    result = []

    for chunk, tokens, masks in zip(chunks, token_embeddings, attention_masks):
        new_chunk = dict(chunk)

        selected = []
        for vec, mask in zip(tokens, masks):
            if mask == 1:
                selected.append(vec)

        if not selected:
            embedding = [0.0] * len(tokens[0])
        else:
            dim = len(selected[0])
            avg = [0.0] * dim
            for vec in selected:
                for d in range(dim):
                    avg[d] += vec[d]

            for d in range(dim):
                avg[d] /= len(selected)

            norm = math.sqrt(sum(v * v for v in avg))

            if norm > 0:
                embedding = [v / norm for v in avg]
            else:
                embedding = [0.0] * dim

        new_chunk["embedding"] = embedding
        result.append(new_chunk)
    
    return result