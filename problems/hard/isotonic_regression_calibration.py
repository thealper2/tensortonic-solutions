import numpy as np


def calibrate_isotonic(cal_labels: list, cal_probs: list, new_probs: list) -> list:
    """
    Returns a list of calibrated probabilities.
    """
    pairs = sorted(zip(cal_probs, cal_labels))
    cal_probs_sorted = [p for p, _ in pairs]
    cal_labels_sorted = [l for _, l in pairs]

    n = len(cal_labels_sorted)

    blocks = []

    for label in cal_labels_sorted:
        blocks.append([label, 1, label])

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][2] > blocks[i + 1][2]:
                sum1, count1, _ = blocks[i]
                sum2, count2, _ = blocks[i + 1]
                new_sum = sum1 + sum2
                new_count = count1 + count2
                new_mean = new_sum / new_count
                blocks[i] = [new_sum, new_count, new_mean]
                del blocks[i + 1]
                merged = True
                if i > 0:
                    i -= 1
            else:
                i += 1

    fitted_probs = []
    fitted_values = []
    for block in blocks:
        sum_labels, count, mean = block
        fitted_probs.extend([cal_probs_sorted[0]] * count)
        fitted_values.extend([mean] * count)

    fitted_values_sorted = []
    idx = 0
    for block in blocks:
        count = block[1]
        fitted_values_sorted.extend([block[2]] * count)

    result = []
    for p in new_probs:
        if p <= cal_probs_sorted[0]:
            result.append(fitted_values_sorted[0])
        elif p >= cal_probs_sorted[-1]:
            result.append(fitted_values_sorted[-1])
        else:
            for i in range(len(cal_probs_sorted) - 1):
                if cal_probs_sorted[i] <= p <= cal_probs_sorted[i + 1]:
                    p1, p2 = cal_probs_sorted[i], cal_probs_sorted[i + 1]
                    v1, v2 = fitted_values_sorted[i], fitted_values_sorted[i + 1]
                    t = (p - p1) / (p2 - p1)
                    result.append(v1 + t * (v2 - v1))
                    break
            else:
                result.append(fitted_values_sorted[-1])

    return result