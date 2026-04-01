def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    num_trees = len(predictions)
    if num_trees == 0:
        return []

    num_samples = len(predictions[0])
    result = []

    for i in range(num_samples):
        votes = {}
        for t in range(num_trees):
            cls = predictions[t][i]
            votes[cls] = votes.get(cls, 0) + 1

        max_votes = max(votes.values())
        winners = [cls for cls, cnt in votes.items() if cnt == max_votes]
        result.append(min(winners))

    return result
    
