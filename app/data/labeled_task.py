from pydantic import BaseModel, Field

from app.environment.state import Action
from app.environment.task import Task


class LabeledTask(BaseModel):
    template: str = Field(description="Name of the task-bank template that generated this task, e.g. 'calculator' or 'mix_k3'.")
    task: Task
    expected_answer: str
    expected_actions: list[Action]
