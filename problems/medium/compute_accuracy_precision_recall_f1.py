import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError()

    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    support = cm.sum(axis=1)
    accuracy = TP.sum() / cm.sum()

    precision_c = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=TP + FP != 0)
    recall_c = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=TP + FN != 0)
    f1_c = np.divide(2 * precision_c * recall_c, precision_c + recall_c, out=np.zeros_like(2 * precision_c * recall_c, dtype=float), where=precision_c + recall_c != 0)

    if average == "micro":
        TP_g, FP_g, FN_g = TP.sum(), FP.sum(), FN.sum()
        precision = TP_g / (TP_g + FP_g) if (TP_g + FP_g) > 0 else 0.0
        recall    = TP_g / (TP_g + FN_g) if (TP_g + FN_g) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    elif average == "macro":
        precision = precision_c.mean()
        recall = recall_c.mean()
        f1 = f1_c.mean()

    elif average == "weighted":
        weights = support / support.sum()
        precision = (precision_c * weights).sum()
        recall = (recall_c * weights).sum()
        f1 = (f1_c * weights).sum()

    elif average == "binary":
        if pos_label is None:
            raise ValueError()

        if pos_label not in label_to_idx:
            raise ValueError()

        i = label_to_idx[pos_label]
        precision = precision_c[i]
        recall = recall_c[i]
        f1 = f1_c[i]

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
