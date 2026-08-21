def parse_react_step(text):
    """
    Returns: parsed thought, action, action input, and final answer
    """
    result = {
        'thought': None,
        'action': None,
        'action_input': None,
        'final_answer': None,
    }

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith('Thought:'):
            result['thought'] = stripped[8:].strip()
        elif stripped.startswith('Action:'):
            result['action'] = stripped[7:].strip()
        elif stripped.startswith('Action Input:'):
            result['action_input'] = stripped[13:].strip()
        elif stripped.startswith('Final Answer:'):
            result['final_answer'] = stripped[13:].strip()

    return result