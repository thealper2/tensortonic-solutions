import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    pmf = (comb(n, k, exact=True) * (p ** k) * ((1 - p) ** (n - k)))
    cdf = sum((comb(n, i, exact=True) * (p ** i) * ((1 - p) ** (n - i))) for i in range(k + 1))
    return float(pmf), float(cdf)
