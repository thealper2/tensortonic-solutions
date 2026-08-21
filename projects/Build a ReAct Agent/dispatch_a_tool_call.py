def invoke_tool_call(action, action_input, tools):
    """
    Returns: the string observation produced by actually calling the requested tool
    """
    if action in tools:
        return tools[action](action_input)
    else:
        return f"Unknown tool: {action}"
