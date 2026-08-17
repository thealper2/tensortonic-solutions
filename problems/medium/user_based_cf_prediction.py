def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    a, b = 0, 0
    for similarity, rating in zip(similarities, ratings):
        if similarity < 0:
            continue

        a += similarity * rating
        b += similarity

    return a / b if b > 0 else 0.0