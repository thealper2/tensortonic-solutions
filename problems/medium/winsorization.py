import numpy as np

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    if not values:
        return []

    n = len(values)
    values = sorted(values)
    values = np.asarray(values)
    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)
    return [float(max(lower, min(upper, v))) for v in values]
