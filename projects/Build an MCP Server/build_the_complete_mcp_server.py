import copy
import json

PROTOCOL_VERSION = "2026-07-28"
SERVER_INFO = {"name": "support-operations", "version": "1.0.0"}

SUPPORT_DOCS = {
    "returns": "Returns are accepted within 30 days of delivery.",
    "shipping": "Standard shipping takes 3 to 5 business days.",
    "warranty": "Hardware includes a one-year limited warranty.",
}

ORDER_STATUS = {
    "A100": {"status": "shipped", "eta": "2026-08-12"},
    "A101": {"status": "processing", "eta": None},
    "A102": {"status": "delivered", "eta": "2026-08-06"},
}

def search_support_docs(query: str) -> dict:
    terms = set(query.lower().split())
    matches = [
        {"topic": topic, "text": text}
        for topic, text in SUPPORT_DOCS.items()
        if topic in terms or terms.intersection(text.lower().split())
    ]
    return {"matches": matches}

def get_order_status(order_id: str) -> dict:
    order = ORDER_STATUS.get(order_id.upper())
    return order or {"status": "not_found", "eta": None}

print(f"Application functions ready for {SERVER_INFO['name']}.")

TOOLS = [
    {
        "name": "search_support_docs",
        "title": "Search Support Documents",
        "description": "Search support documents for a given query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "annotations": {
            "readOnly": True
        }
    },
    {
        "name": "get_order_status",
        "title": "Get Order Status",
        "description": "Get the status of an order by order ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
            "additionalProperties": False
        },
        "annotations": {
            "readOnly": True
        }
    }
]

TOOL_HANDLERS = {
    "search_support_docs": search_support_docs,
    "get_order_status": get_order_status,
}

print(json.dumps({"tools": TOOLS}, indent=2))

RESOURCE_DATA = {
    "support://policies/all": {
        "name": "support-policies",
        "title": "Complete support policies",
        "description": "Current return, shipping, and warranty policies.",
        "mimeType": "text/plain",
        "text": "\n".join(SUPPORT_DOCS.values()),
    }
}

RESOURCES = [
    {
        "name": data["name"],
        "title": data["title"],
        "description": data["description"],
        "uri": uri,
        "mimeType": data["mimeType"]
    }
    for uri, data in RESOURCE_DATA.items()
]

def read_resource(uri: str) -> dict:
    if uri not in RESOURCE_DATA:
        raise KeyError(f"Resource not found: {uri}")
    data = RESOURCE_DATA[uri]
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": data["mimeType"],
                "text": data["text"]
            }
        ]
    }


print(json.dumps({"resources": RESOURCES}, indent=2))

JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}

def matches_json_type(value, type_name: str) -> bool:
    if type_name == "integer":
        return type(value) is int
    if type_name == "number":
        return type(value) in (int, float)
    expected = JSON_TYPES.get(type_name)
    return expected is not None and isinstance(value, expected)

def validate_arguments(arguments: dict, input_schema: dict) -> dict:
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    missing = []
    unknown = []
    wrong_type = []

    for field in required:
        if field not in arguments:
            missing.append(field)

    for key, value in arguments.items():
        if key not in properties:
            if not additional_properties:
                unknown.append(key)
            continue

        schema_type = properties[key].get("type")
        if schema_type and not matches_json_type(value, schema_type):
            wrong_type.append(key)

    errors = {
        "missing": missing,
        "unknown": unknown,
        "wrong_type": wrong_type,
    }
    return {"valid": not any(errors.values()), "errors": errors}


sample_schema = TOOLS[0]["inputSchema"]
print(validate_arguments({"query": "returns"}, sample_schema))
print(validate_arguments({"query": 42, "extra": True}, sample_schema))

def call_tool(name: str, arguments: dict) -> dict:
    descriptor = next((tool for tool in TOOLS if tool["name"] == name), None)
    handler = TOOL_HANDLERS.get(name)

    if descriptor is None or handler is None:
        raise LookupError(f"Unknown tool: {name}")

    validation = validate_arguments(arguments, descriptor["inputSchema"])
    if not validation["valid"]:
        message = json.dumps(validation["errors"])
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }

    result = handler(**arguments)
    if result is None:
        raise ValueError("Execute the registered tool before returning its result.")

    return {
        "content": [{"type": "text", "text": json.dumps(result)}],
        "structuredContent": result,
        "isError": False,
    }


print(json.dumps(call_tool("search_support_docs", {"query": "returns"}), indent=2))

def success(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

def failure(request_id, code: int, message: str, data=None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}

def handle_request(request: dict) -> dict:
    request_id = request.get("id")
    if request.get("jsonrpc") != "2.0" or "method" not in request:
        return failure(request_id, -32600, "Invalid Request")

    method = request["method"]
    params = request.get("params", {})

    if method == "server/discover":
        return success(request_id, {
            "resultType": "server_discovery",
            "supportedVersions": [PROTOCOL_VERSION],
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "resources": {
                    "listChanged": False
                }
            },
            "serverInfo": SERVER_INFO
        })

    elif method == "tools/list":
        return success(request_id, {"tools": TOOLS})

    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = call_tool(name, arguments)
            return success(request_id, result)
        except LookupError:
            return failure(request_id, -32602, f"Unknown tool: {name}")
        except Exception as e:
            return failure(request_id, -32603, str(e))

    elif method == "resources/list":
        return success(request_id, {"resources": RESOURCES})

    elif method == "resources/read":
        uri = params.get("uri")
        try:
            result = read_resource(uri)
            return success(request_id, result)
        except KeyError:
            return failure(request_id, -32002, f"Resource not found: {uri}")

    else:
        return failure(request_id, -32601, f"Method not found: {method}")

CLIENT_META = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
    "io.modelcontextprotocol/clientInfo": {
        "name": "support-console",
        "version": "1.0.0",
    },
    "io.modelcontextprotocol/clientCapabilities": {},
}

def request(request_id, method, **params):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**params, "_meta": CLIENT_META},
    }

requests = [
    request(1, "server/discover"),
    request(2, "tools/list"),
    request(3, "resources/list"),
    request(4, "resources/read", uri="support://policies/all"),
    request(5, "tools/call", name="get_order_status", arguments={"order_id": "A100"}),
    request(6, "tools/call", name="search_support_docs", arguments={"query": 42}),
]

responses = []
for client_request in requests:
    response = handle_request(copy.deepcopy(client_request))
    responses.append(response)
    print(f"CLIENT  {client_request['method']}  id={client_request['id']}")
    print("SERVER ", json.dumps(response, ensure_ascii=False))
    print()

checks = {
    "preserved_ids": all(response["id"] == index for index, response in enumerate(responses, start=1)),
    "listed_tools": responses[1]["result"]["tools"] == TOOLS,
    "read_resource": bool(responses[3]["result"]["contents"]),
    "called_tool": responses[4]["result"]["isError"] is False,
    "rejected_bad_arguments": responses[5]["result"]["isError"] is True,
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")