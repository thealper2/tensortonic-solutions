def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    seen_keys = {}
    result = []
    
    for record in records:
        key = tuple(record[col] for col in key_columns)
        
        if key not in seen_keys:
            seen_keys[key] = record
            result.append(record)
        else:
            if strategy == "first":
                continue
            elif strategy == "last":
                idx = result.index(seen_keys[key])
                result[idx] = record
                seen_keys[key] = record
            elif strategy == "most_complete":
                current_none_count = sum(1 for val in record.values() if val is None)
                existing_record = seen_keys[key]
                existing_none_count = sum(1 for val in existing_record.values() if val is None)
                
                if current_none_count < existing_none_count:
                    idx = result.index(existing_record)
                    result[idx] = record
                    seen_keys[key] = record
    
    return result
