import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    auc = np.trapezoid(tpr, fpr)
    return auc
