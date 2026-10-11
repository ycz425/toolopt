from app.agents.base import Policy
from app.agents.gemini import GeminiAgent
from app.agents.prompting import format_history, format_tools
from app.environment.task import Task
from app.tools.base import ToolMetadata
from app.environment.state import State, Action
from datetime import datetime


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