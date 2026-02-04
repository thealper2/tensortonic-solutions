def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    unique_items = []
    for r in recommendations:
        for c in r:
            if c not in unique_items:
                unique_items.append(c)

    return len(unique_items) / n_items
