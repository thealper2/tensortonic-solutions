from datetime import datetime

def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    best_model = sorted(
        models,
        key=lambda m: (
            -m["accuracy"],
            m["latency"],
            -datetime.fromisoformat(m["timestamp"]).timestamp(),
        )
    )[0]
    return best_model["name"]
