from pydantic import BaseModel, Field
from app.tools.base import ToolMetadata
from app.tools.executor import ToolExecutionResult
from app.environment.task import Task
from typing import Any


class Action(BaseModel):
    tool_name: str = Field(description="Name of the tool that was called.")
    args: dict[str, Any] = Field(description="Arguments the tool was called with.")


class Step(BaseModel):
    action: Action = Field(description="The tool call that was made.")
    result: ToolExecutionResult = Field(description="The outcome of executing the action.")


class State(BaseModel):
    task: Task = Field(description="The task the agent is trying to complete.")
    available_tools: list[ToolMetadata] = Field(description="Tools the agent may choose from at this point.")
    history: list[Step] = Field(default_factory=list, description="Steps taken so far this episode, in order.")
