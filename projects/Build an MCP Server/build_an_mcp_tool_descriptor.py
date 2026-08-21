def build_mcp_tool_descriptor(name, description, properties, required):
    """
    Returns: a discoverable MCP tool descriptor
    """
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    }