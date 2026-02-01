import numpy as np

def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    n = len(s1)
    m = len(s2)
    dp = np.zeros((n + 1,m + 1))

    for i in range(n + 1) :
        dp[i, 0] = i 
    for i in range(m + 1) :
        dp[0, i] = i

    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + (1 if s1[i - 1]!= s2[j - 1] else 0)
            )

    return int(dp[n, m])
