def build_jsonrpc_request(request_id, method, params):
    """
    Returns: one JSON-RPC request dictionary
    """
    request = {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': method,
    }

    if params is not None:
        request['params'] = params

    return request