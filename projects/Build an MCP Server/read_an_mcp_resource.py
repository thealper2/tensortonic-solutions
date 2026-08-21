def read_mcp_resource(uri, resources):
    """
    Returns: whether the resource exists and its MCP contents
    """
    if uri in resources:
        resource = resources[uri]
        return {
            "found": True,
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource["mimeType"],
                    "text": resource["text"]
                }
            ]
        }
    else:
        return {
            "found": False,
            "contents": []
        }