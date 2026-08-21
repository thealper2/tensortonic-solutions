def dispatch_mcp_tool(name, arguments, documents, orders):
    """
    Returns: an MCP tool result with typed content
    """
    if name == "search_docs":
        query = arguments.get("query", "").lower()
        for key, value in documents.items():
            if query in key.lower():
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": value}]
                }
        return {
            "isError": False,
            "content": [{"type": "text", "text": "No matching document found."}]
        }

    elif name == "get_order_status":
        order_id = arguments.get("order_id")
        if order_id in orders:
            return {
                "isError": False,
                "content": [{"type": "text", "text": orders[order_id]}]
            }
        return {
            "isError": False,
            "content": [{"type": "text", "text": "Order not found."}]
        }

    else:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}]
        }