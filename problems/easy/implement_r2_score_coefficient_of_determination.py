import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.all(y_true == y_true[0]):
      return 1.0 if np.all(y_pred == y_true) else 0.0

    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return float(r2)
