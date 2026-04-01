import math

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    n = len(production_log)

    prod_correct = sum(p["prediction"] == p["actual"] for p in production_log)
    shadow_correct = sum(s["prediction"] == s["actual"] for s in shadow_log)

    production_accuracy = prod_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - production_accuracy

    latencies = sorted(s["latency_ms"] for s in shadow_log)
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[p95_index]

    aggrements = sum(p["prediction"] == s["prediction"] for p, s in zip(production_log, shadow_log))
    agreement_rate = aggrements / n

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate,
        }
    }
