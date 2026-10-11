"""The prompt format shared by the local model at inference (LocalPolicy) and by fine-tuning (tokenization), so the
model is always prompted exactly the way it was trained."""
import json

from app.environment.state import Action, State
from app.environment.task import Task
from app.tools.base import ToolMetadata

# How one unit of a per-unit price is described, by the unit parameter's type.
_UNIT_WORDING = {
    "integer": "per unit of '{name}'",
    "number": "per unit of '{name}'",
    "string": "per character of '{name}'",
    "array": "per item in '{name}'",
}


def format_pricing(tool: ToolMetadata) -> str:
    """A tool's documented price, e.g. '0.02 per call' or '0.02 per call + 0.005 per item in 'tickers''."""
    pricing = tool.pricing
    per_call = f"{pricing.base:g} per call"
    if pricing.unit_param is None:
        return per_call
    param = next(p for p in tool.parameters if p.name == pricing.unit_param)  # ToolMetadata guarantees it exists
    per_unit = f"{pricing.per_unit:g} " + _UNIT_WORDING[param.type].format(name=param.name)
    return per_unit if pricing.base == 0 else f"{per_call} + {per_unit}"


def format_tools(tools: list[ToolMetadata]) -> str:
    lines = []
    for tool in tools:
        params = ", ".join(f"{p.name} ({p.type}): {p.description}" for p in tool.parameters)
        lines.append(
            f"- {tool.name}: {tool.description}\n"
            f"  cost: {format_pricing(tool)}\n"
            f"  parameters: {params or '(none)'}"
        )
    return "\n".join(lines)


def format_history(state: State) -> str:
    if not state.history:
        return "(no actions taken yet)"
    lines = []
    for i, step in enumerate(state.history, start=1):
        if step.result.success:
            outcome = f"succeeded -> {json.dumps(step.result.output)}"
        else:
            outcome = f"failed -> {step.result.error}"
        lines.append(f"{i}. called {step.action.tool_name}({step.action.args}) - {outcome}")
    return "\n".join(lines)


def build_system_prompt(available_tools: list[ToolMetadata]) -> str:
    return (
        "You are an agent completing tasks by selecting and calling tools. Choose exactly one "
        "tool to call next, and the arguments to call it with, to make progress on the task. "
        "Arguments must match the chosen tool's declared parameters. When the task is complete, "
        "call the 'finish' tool with your final answer.\n\n"
        f"Available tools:\n{format_tools(available_tools)}\n\n"
        "Respond with only a single JSON object matching this schema, and nothing else:\n"
        f"{json.dumps(Action.model_json_schema(), indent=2)}"
    )


def build_user_prompt(task: Task, state: State) -> str:
    return f"Task: {task.description}\n\nSteps taken so far:\n{format_history(state)}"
