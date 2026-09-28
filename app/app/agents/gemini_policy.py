import json

from app.agents.base import GeminiAgent, Policy
from app.environment.task import Task
from app.tools.base import ToolMetadata
from app.environment.state import State, Action
from datetime import datetime


def format_tools(tools: list[ToolMetadata]) -> str:
    lines = []
    for tool in tools:
        params = ", ".join(f"{p.name} ({p.type}): {p.description}" for p in tool.parameters)
        lines.append(f"- {tool.name}: {tool.description}\n  parameters: {params or '(none)'}")
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


def build_prompt(task: Task, tools: list[ToolMetadata], state: State) -> str:
    return (
        f"You are an agent completing the following task:\n{task.description}\n\n"
        f"Available tools:\n{format_tools(tools)}\n\n"
        f"Steps taken so far:\n{format_history(state)}\n\n"
        "Choose exactly one tool to call next, and the arguments to call it with, "
        "to make progress on the task. Arguments must match the chosen tool's "
        "declared parameters. When the task is complete, call the 'finish' tool "
        "with your final answer."
    )


class GeminiPolicy(GeminiAgent, Policy):
    label = 'gemini_policy'
    async def select_action(self, task: Task, tools: list[ToolMetadata], state: State, temperature: float = 0) -> Action:
        if self.verbose:
            print(f'{datetime.now()}     Selecting action...')
        return await super().generate_structured(
            prompt=build_prompt(task, tools, state),
            schema=Action,
            label=self.label,
            temperature=temperature
        )