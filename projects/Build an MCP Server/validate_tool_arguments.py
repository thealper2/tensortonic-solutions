def validate_tool_arguments(arguments, input_schema):
    """
    Returns: argument validation details
    """
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    additional_properties = input_schema.get("additionalProperties", True)

    type_map = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    missing = []
    unknown = []
    invalid_types = []

    for field in required:
        if field not in arguments:
            missing.append(field)

    for key, value in arguments.items():
        if key not in properties:
            if not additional_properties:
                unknown.append(key)
            continue

        schema_type = properties[key].get("type")
        if schema_type in type_map:
            if schema_type == "number" and isinstance(value, bool):
                invalid_types.append(key)
                continue

            if not isinstance(value, type_map[schema_type]):
                invalid_types.append(key)

    valid = len(missing) == 0 and len(unknown) == 0 and len(invalid_types) == 0

    return {
        "valid": valid,
        "missing": missing,
        "unknown": unknown,
        "invalid_types": invalid_types,
    }