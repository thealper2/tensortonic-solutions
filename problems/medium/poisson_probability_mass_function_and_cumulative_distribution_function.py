import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    log_fact = np.zeros(k + 1)
    for i in range(1, k + 1):
        log_fact[i] = log_fact[i - 1] + np.log(i)

    log_pmf_all = -lam + np.arange(k + 1) * np.log(lam) - log_fact
    pmf = np.exp(log_pmf_all[k])
    cdf = np.sum(np.exp(log_pmf_all[:k + 1]))
    return float(pmf), float(cdf)