def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0]) if n_users > 0 else 0
    
    ratings = []
    for i in range(n_users):
        for j in range(n_items):
            if ratings_matrix[i][j] != 0:
                ratings.append(ratings_matrix[i][j])
    
    mu = sum(ratings) / len(ratings)
    
    user_biases = [0.0] * n_users
    for i in range(n_users):
        user_ratings = [r for r in ratings_matrix[i] if r != 0]
        if user_ratings:
            user_biases[i] = sum(user_ratings) / len(user_ratings) - mu
    
    item_biases = [0.0] * n_items
    for j in range(n_items):
        item_ratings = [ratings_matrix[i][j] for i in range(n_users) if ratings_matrix[i][j] != 0]
        if item_ratings:
            item_biases[j] = sum(item_ratings) / len(item_ratings) - mu
    
    predictions = []
    for user, item in target_pairs:
        pred = mu + user_biases[user] + item_biases[item]
        predictions.append(pred)
    
    return predictions