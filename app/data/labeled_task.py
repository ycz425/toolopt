from pydantic import BaseModel, Field

from app.environment.state import Action
from app.environment.task import Task


class LabeledTask(BaseModel):
    template: str = Field(description="Name of the task-bank template that generated this task, e.g. 'calculator' or 'mix_k3'.")
    task: Task
    expected_answer: str
    expected_actions: list[Action] = Field(description="The cheapest valid path (ending with finish). generate_data checks it is the cheapest.")
    alternative_paths: list[list[Action]] = Field(
        default_factory=list,
        description="Other valid paths that reach the same answer, usually at a higher cost (each ending with finish).",
    )
    invalid_paths: list[list[Action]] = Field(
        default_factory=list,
        description="Plausible paths that fail, e.g. a cheap tool used outside its coverage. Used only as rejected DPO examples.",
    )

    @property
    def valid_paths(self) -> list[list[Action]]:
        """Every valid path, cheapest (expected_actions) first."""
        return [self.expected_actions, *self.alternative_paths]
