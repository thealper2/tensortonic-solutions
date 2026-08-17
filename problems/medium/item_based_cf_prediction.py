def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    n = len(user_ratings)
    a, b = 0, 0
    for i, rating in enumerate(user_ratings):
        if i == target or rating == 0:
            continue
            
        similarity = item_similarities[i]
        if similarity < 0:
            continue
        
        a += rating * similarity
        b += similarity

    return a / b if b > 0 else 0.0