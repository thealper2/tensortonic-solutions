def evaluate_agent_trajectory(actual_calls, expected_calls, mode, max_steps):
    """
    Returns: trajectory match details and the final pass decision
    """
    within_budget = len(actual_calls) <= max_steps

    if mode == "strict":
        matched_steps = 0
        for i in range(min(len(actual_calls), len(expected_calls))):
            if actual_calls[i].get("action") == expected_calls[i].get("action") and \
               actual_calls[i].get("action_input") == expected_calls[i].get("action_input"):
                matched_steps += 1
            else:
                break

        passed = matched_steps == len(expected_calls) and len(actual_calls) == len(expected_calls)
        extra_steps = max(0, len(actual_calls) - matched_steps)

    elif mode == "ordered_subset":
        expected_set = set()
        for call in expected_calls:
            expected_set.add((call.get("action"), call.get("action_input")))

        matched_steps = 0
        expected_idx = 0
        matched_positions = set()

        for i, actual in enumerate(actual_calls):
            if expected_idx < len(expected_calls):
                if actual.get("action") == expected_calls[expected_idx].get("action") and \
                   actual.get("action_input") == expected_calls[expected_idx].get("action_input"):
                    matched_steps += 1
                    matched_positions.add(i)
                    expected_idx += 1

        extra_steps = 0
        for i, actual in enumerate(actual_calls):
            if i not in matched_positions:
                actual_key = (actual.get("action"), actual.get("action_input"))
                if actual_key not in expected_set:
                    extra_steps += 1

        passed = matched_steps == len(expected_calls)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    passed = passed and within_budget

    return {
        "passed": passed,
        "matched_steps": matched_steps,
        "required_steps": len(expected_calls),
        "extra_steps": extra_steps,
        "within_budget": within_budget
    }