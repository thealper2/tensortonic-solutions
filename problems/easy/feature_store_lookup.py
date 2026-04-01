def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    result = []
    for request in requests:
        user_id = request["user_id"]
        online_features = request["online_features"]

        if user_id in feature_store:
            offline_features = feature_store[user_id]
        else:
            offline_features = defaults.copy()

        combined = {}
        combined.update(offline_features)
        combined.update(online_features)
        result.append(combined)

    return result
