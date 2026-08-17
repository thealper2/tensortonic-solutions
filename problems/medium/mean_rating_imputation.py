def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    result = [row[:] for row in ratings_matrix]
    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0]) if n_users > 0 else 0

    if mode == "user":
        for i in range(n_users):
            non_zero = [x for x in ratings_matrix[i] if x != 0]
            if non_zero:
                mean_val = sum(non_zero) / len(non_zero)
                for j in range(n_items):
                    if result[i][j] == 0:
                        result[i][j] = mean_val

    elif mode == "item":
        for j in range(n_items):
            non_zero = []
            for i in range(n_users):
                if ratings_matrix[i][j] != 0:
                    non_zero.append(ratings_matrix[i][j])

            if non_zero:
                mean_val = sum(non_zero) / len(non_zero)
                for i in range(n_users):
                    if result[i][j] == 0:
                        result[i][j] = mean_val

    return result