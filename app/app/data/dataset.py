from pathlib import Path

from pydantic import BaseModel
from app.data.trajectory import Trajectory
from app.environment.state import Action, Step
from app.environment.task import Task
from app.tools.base import ToolMetadata


class TrainingExample(BaseModel):
    task: Task
    available_tools: list[ToolMetadata]
    history: list[Step]
    target_action: Action


def trajectory_to_examples(trajectory: Trajectory) -> list[TrainingExample]:
    examples = []
    for i, step in enumerate(trajectory.steps):
        examples.append(TrainingExample(
            task=trajectory.task,
            available_tools=trajectory.available_tools,
            history=trajectory.steps[:i],
            target_action=step.action
        ))
    return examples


def load_trajectories(path: str | Path) -> list[Trajectory]:
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(Trajectory.model_validate_json(line))
    return trajectories


def build_dataset(path: str | Path, include_failed: bool = False) -> list[TrainingExample]:
    examples = []
    for trajectory in load_trajectories(path):
        if not trajectory.success and not include_failed:
            continue
        examples.extend(trajectory_to_examples(trajectory))
    return examples
