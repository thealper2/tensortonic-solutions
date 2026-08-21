def authorize_tool_call(tool_name, arguments, policies, approved):
    """
    Returns: permission decision, reason, and missing arguments
    """
    if tool_name not in policies:
        return {
            'allowed': False,
            'reason': 'unknown_tool',
            'missing': [],
        }

    policy = policies[tool_name]
    required_fields = policy.get('required_fields', [])

    missing = []
    for field in required_fields:
        if field not in arguments:
            missing.append(field)

    if missing:
        return {
            'allowed': False,
            'reason': 'missing_arguments',
            'missing': missing,
        }

    if policy.get('requires_approval', False) and not approved:
        return {
            'allowed': False,
            'reason': 'approval_required',
            'missing': [],
        }

    return {
        'allowed': True,
        'reason': 'allowed',
        'missing': [],
    }