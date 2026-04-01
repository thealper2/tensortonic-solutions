def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    result = []
    for avg_rating, num_votes in items:
        wr = (num_votes / (num_votes + min_votes)) * avg_rating + (min_votes / (num_votes + min_votes)) * global_mean
        result.append(wr)

    return result
