def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    n = len(rater1)
    aggrements = sum(a == b for a, b in zip(rater1, rater2))
    p_o = aggrements / n

    freq1 = {}
    freq2 = {}
    unique_labels = set()

    for i in rater1:
        if i in freq1:
            freq1[i] += 1
        else:
            freq1[i] = 1

        unique_labels.add(i)

    for i in rater2:
        if i in freq2:
            freq2[i] += 1
        else:
            freq2[i] = 1

        unique_labels.add(i)

    p_e = 0.0
    for label in unique_labels:
        p_e += (freq1.get(label, 0) / n) * (freq2.get(label, 0) / n)

    if p_e == 1:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa
