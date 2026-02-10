import numpy as np

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    ratings = np.asarray(ratings_matrix, dtype=float)
    n = ratings.shape[0]
    user_means = np.zeros(n)
    for u in range(n):
        rated = ratings[u] != 0
        if np.any(rated):
            user_means[u] = ratings[u, rated].mean()
    
    numerator = 0.0
    denom_i = 0.0
    denom_j = 0.0

    for u in range(n):
        r_ui = ratings[u, item_i]
        r_uj = ratings[u, item_j]

        if r_ui != 0 and r_uj != 0:
            ci = r_ui - user_means[u]
            cj = r_uj - user_means[u]

            numerator += ci * cj
            denom_i += ci ** 2
            denom_j += cj ** 2

    denominator = np.sqrt(denom_i) * np.sqrt(denom_j)

    if denominator == 0:
        return 0.0

    return numerator / denominator
