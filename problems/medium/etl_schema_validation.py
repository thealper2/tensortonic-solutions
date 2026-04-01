def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    results = []
    for idx, record in enumerate(records):
        errors = []
        for col_def in schema:
            col = col_def["column"]
            expected_type = col_def["type"]
            nullable = col_def.get("nullable", False)
            min_val = col_def.get("min")
            max_val = col_def.get("max")

            if col not in record:
                errors.append(f"{col}: missing")
                continue

            value = record[col]

            if value is None:
                if not nullable:
                    errors.append(f"{col}: null")
                
                continue

            actual_type = type(value).__name__

            if expected_type == "float":
                if type(value) not in (int, float):
                    errors.append(f"{col}: expected float, got {actual_type}")
                    continue
            else:
                if actual_type != expected_type:
                    errors.append(f"{col}: expected {expected_type}, got {actual_type}")
                    continue

            if min_val is not None or max_val is not None:
                if isinstance(value, (int, float)):
                    if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                        errors.append(f"{col}: out of range")

        results.append((idx, len(errors) == 0, errors))

    return results
