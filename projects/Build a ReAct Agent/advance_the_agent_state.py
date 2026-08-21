def advance_agent_state(state, step, observation):
    """
    Returns: a new state after one model decision
    """
    new_state = {
        "status": state["status"],
        "scratchpad": state["scratchpad"],
        "trajectory": state["trajectory"].copy(),
        "final_answer": state["final_answer"]
    }

    if step.get("final_answer") is not None:
        new_state["status"] = "completed"
        new_state["final_answer"] = step["final_answer"]
        new_state["scratchpad"] += f"Thought: {step.get('thought', '')}\nFinal Answer: {step['final_answer']}"
        return new_state

    thought = step.get("thought", "")
    action = step.get("action")
    action_input = step.get("action_input", "")

    new_state["scratchpad"] += f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"

    new_state["trajectory"].append({
        "action": action,
        "action_input": action_input,
        "observation": observation
    })

    new_state["status"] = "running"
    new_state["final_answer"] = None

    return new_state