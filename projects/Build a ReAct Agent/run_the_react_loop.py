def run_react_loop(steps, max_steps, tools):
    """
    Returns: loop status, final answer, and tool trajectory
    """
    trajectory = []

    for step in steps[:max_steps]:
        if step.get("final_answer") is not None:
            return {
                "status": "completed",
                "answer": step["final_answer"],
                "trajectory": trajectory
            }

        action = step.get("action")
        action_input = step.get("action_input")

        if not action or action_input is None:
            return {
                "status": "invalid_step",
                "answer": None,
                "trajectory": trajectory
            }

        if action in tools:
            observation = tools[action](action_input)
        else:
            observation = f"Unknown tool: {action}"

        trajectory.append({
            "action": action,
            "action_input": action_input,
            "observation": observation
        })

    return {
        "status": "max_steps",
        "answer": None,
        "trajectory": trajectory
    }