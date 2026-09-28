from pydantic import BaseModel, Field

from app.environment.state import State, Step
from app.environment.task import Task
from app.evaluation.metrics import total_cost, total_latency
from app.tools.base import ToolMetadata


class Trajectory(BaseModel):
    task: Task = Field(description="The task this trajectory was generated for.")
    available_tools: list[ToolMetadata] = Field(description="Tools available during this episode.")
    steps: list[Step] = Field(description="The sequence of tool calls and results taken during the episode.")
    success: bool = Field(description="Whether the task was completed successfully.")
    total_cost: float = Field(description="Sum of the cost of every step in the trajectory.")
    total_latency: float = Field(description="Sum of the latency of every step in the trajectory.")

    @classmethod
    def from_state(cls, state: State, success: bool) -> "Trajectory":
        return cls(
            task=state.task,
            available_tools=state.available_tools,
            steps=state.history,
            success=success,
            total_cost=total_cost(state),
            total_latency=total_latency(state),
        )
