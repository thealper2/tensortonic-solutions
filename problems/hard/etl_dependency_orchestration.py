def schedule_pipeline(tasks: list, resource_budget: int) -> list:
    """
    Returns a list of schedule dictionaries.
    """
    task_by_name = {task["name"]: task for task in tasks}
    dependencies = {task["name"]: set(task.get("depends_on", [])) for task in tasks}
    resources = {task["name"]: task["resources"] for task in tasks}
    duration = {task["name"]: task["duration"] for task in tasks}

    completion_time = {task["name"]: None for task in tasks}
    scheduled = []
    running = {}
    time = 0

    completed = set()

    ready = []

    while len(completed) < len(tasks):
        if running:
            next_completion = min(running.values())
            if next_completion > time:
                time = next_completion

            to_complete = [name for name, end in running.items() if end == time]
            for name in to_complete:
                completed.add(name)
                completion_time[name] = time
                del running[name]

        ready_names = []
        for task_name in task_by_name:
            if task_name in completed or task_name in running:
                continue
            if completion_time[task_name] is not None:
                continue
            deps = dependencies[task_name]
            if deps.issubset(completed):
                ready_names.append(task_name)

        ready_names.sort()

        remaining_budget = resource_budget

        started_in_wave = []

        for name in ready_names:
            if name in started_in_wave:
                continue
            if resources[name] <= remaining_budget:
                scheduled.append({"task_name": name, "start_time": time})
                running[name] = time + duration[name]
                remaining_budget -= resources[name]
                started_in_wave.append(name)

        if not running and len(completed) < len(tasks):
            for name in task_by_name:
                if name not in completed and name not in running and completion_time[name] is None:
                    deps = dependencies[name]
                    if deps.issubset(completed):
                        scheduled.append({"task_name": name, "start_time": time})
                        running[name] = time + duration[name]
                        remaining_budget -= resources[name]
                        break

    scheduled.sort(key=lambda x: (x["start_time"], x["task_name"]))
    return scheduled
