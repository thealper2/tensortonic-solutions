import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    if system_type == "classification":
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        n = len(y_true)

        accuracy = (TP + TN) / n if n > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["accuracy"] = accuracy
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    elif system_type == "regression":
        errors = y_true - y_pred
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        metrics["mae"] = mae
        metrics["rmse"] = rmse

    elif system_type == "ranking":
        paired = list(zip(y_true, y_pred))
        paired.sort(key=lambda x: x[1], reverse=True)

        top_k = paired[:3]
        relevant_in_top_k = sum(label for label, _ in top_k)
        total_relevant = np.sum(y_true)

        precision_at_3 = relevant_in_top_k / 3
        recall_at_3 = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        metrics["precision_at_3"] = precision_at_3
        metrics["recall_at_3"] = recall_at_3

    else:
        raise ValueError()

    return sorted(metrics.items(), key=lambda x: x[0])
