from app.agents.gemini_policy import format_history, format_tools
from app.data.dataset import TrainingExample
from app.environment.state import State
from app.environment.task import Task
from app.tools.base import ToolMetadata


def build_system_prompt(available_tools: list[ToolMetadata]) -> str:
    return (
        "You are an agent completing tasks by selecting and calling tools. Choose exactly one "
        "tool to call next, and the arguments to call it with, to make progress on the task. "
        "Arguments must match the chosen tool's declared parameters. When the task is complete, "
        "call the 'finish' tool with your final answer.\n\n"
        f"Available tools:\n{format_tools(available_tools)}"
    )


def build_user_prompt(task: Task, state: State) -> str:
    return f"Task: {task.description}\n\nSteps taken so far:\n{format_history(state)}"


def build_completion(example: TrainingExample) -> str:
    return example.target_action.model_dump_json()
