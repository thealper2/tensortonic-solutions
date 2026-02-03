def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    cum = 1.0
    result = []

    for r in returns:
        cum *= (1 + r)
        result.append(cum - 1)
        
    return result
