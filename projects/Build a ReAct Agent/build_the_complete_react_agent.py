import os
import re
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="google.gemma-3-4b-it",
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0,
)

incident = (
    "Customers in eu-west are receiving HTTP 429 responses from the checkout API. "
    "Check the service status and recommend the next action from the runbook."
)

print(f"Incident: {incident}")

SERVICE_STATUS = {
    "checkout-api": {
        "region": "eu-west",
        "status": "degraded",
        "detail": "Elevated HTTP 429 responses since 14:20 UTC.",
    },
    "identity-api": {
        "region": "global",
        "status": "operational",
        "detail": "No active incident.",
    },
}

RUNBOOKS = {
    "429": (
        "Confirm the affected region, reduce retry concurrency, and use exponential "
        "backoff with jitter. Escalate when the error rate remains elevated for 15 minutes."
    ),
    "latency": (
        "Check regional health, compare p95 latency with the baseline, and inspect recent deploys."
    ),
}

def check_service_status(service: str) -> str:
    status = SERVICE_STATUS.get(service.lower().strip())
    return str(status) if status else "Service not found."

def search_runbook(query: str) -> str:
    normalized = query.lower()
    for topic, guidance in RUNBOOKS.items():
        if topic in normalized:
            return guidance
    return "No matching runbook entry found."

TOOL_REGISTRY = {
    "check_service_status": check_service_status,
    "search_runbook": search_runbook,
}

print(f"Registered tools: {', '.join(TOOL_REGISTRY)}")

def parse_react_step(text: str) -> dict:
    result = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    thought_match = re.search(r"Thought:\s*(.+)", text, re.IGNORECASE)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).strip()

    action_input_match = re.search(r"Action Input:\s*(.+)", text, re.IGNORECASE)
    if action_input_match:
        result["action_input"] = action_input_match.group(1).strip()

    final_answer_match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if final_answer_match:
        result["final_answer"] = final_answer_match.group(1).strip()

    return result


sample = """Thought: I should check the affected service.
Action: check_service_status
Action Input: checkout-api"""

print(parse_react_step(sample))

REACT_SYSTEM_PROMPT = """You are a support operations agent.
Investigate the incident using the available read-only tools before answering.

Available tools:
- check_service_status: accepts a service name such as checkout-api.
- search_runbook: accepts a support topic such as HTTP 429.

For a tool call, respond exactly as:
Thought: <brief reason for the next action>
Action: <tool name>
Action Input: <single tool argument>

After receiving an Observation, either call another tool or finish with:
Thought: <brief conclusion>
Final Answer: <grounded operational response>

Do not invent service status or runbook guidance.
"""

print(REACT_SYSTEM_PROMPT)

def run_support_agent(question: str, max_steps: int = 5) -> dict:
    scratchpad = f"Question: {question}"
    trace = []

    for step_number in range(1, max_steps + 1):
        messages = [
            ("system", REACT_SYSTEM_PROMPT),
            ("human", scratchpad),
        ]
        model_reply = llm.invoke(messages).content
        if model_reply is None:
            raise ValueError("Invoke the model before parsing its decision.")

        decision = parse_react_step(model_reply)
        if decision is None:
            raise ValueError("Parse the model response before continuing.")

        event = {
            "step": step_number,
            "model_reply": model_reply,
            "action": decision["action"],
            "action_input": decision["action_input"],
        }

        if decision["final_answer"]:
            event["observation"] = None
            trace.append(event)
            return {
                "status": "completed",
                "answer": decision["final_answer"],
                "trace": trace,
            }

        tool = TOOL_REGISTRY.get(decision["action"])
        if tool is None:
            observation = f"Unknown tool: {decision['action']}"
        else:
            observation = tool(decision["action_input"])

        if observation is None:
            raise ValueError("Execute the selected tool before continuing.")

        event["observation"] = observation
        trace.append(event)
        scratchpad += f"\n{model_reply}\nObservation: {observation}"

    return {
        "status": "step_limit_reached",
        "answer": "The agent could not complete the investigation within the step limit.",
        "trace": trace,
    }


print("Agent loop ready.")

result = run_support_agent(incident)

for event in result["trace"]:
    print(f"Step {event['step']}")
    print(f"Action: {event['action'] or 'finish'}")
    if event["action_input"]:
        print(f"Input: {event['action_input']}")
    if event["observation"]:
        print(f"Observation: {event['observation']}")
    print()

print(f"Status: {result['status']}")
print(f"Answer: {result['answer']}")

tool_calls = [
    event["action"]
    for event in result["trace"]
    if event["action"] in TOOL_REGISTRY
]

checks = {
    "completed": result["status"] == "completed",
    "used_registered_tool": len(tool_calls) > 0,
    "within_step_budget": len(result["trace"]) <= 5,
    "returned_answer": bool(result["answer"].strip()),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

print(f"\nTrajectory score: {sum(checks.values())}/{len(checks)}")