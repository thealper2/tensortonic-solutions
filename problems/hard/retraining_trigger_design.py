def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    last_retrain_day = -cooldown
    days_since_retrain = 0
    retraining_days = []

    for stat in daily_stats:
        day = stat["day"]
        days_since_retrain += 1

        drift_trigger = stat["drift_score"] > drift_threshold
        performance_trigger = stat["performance"] < performance_threshold
        staleness_trigger = days_since_retrain >= max_staleness

        cooldown_ok = day - last_retrain_day >= cooldown
        budget_ok = budget >= retrain_cost

        if (drift_trigger or performance_trigger or staleness_trigger) and cooldown_ok and budget_ok:
            retraining_days.append(day)
            last_retrain_day = day
            days_since_retrain = 0
            budget -= retrain_cost

    return retraining_days