from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field

from app.environment.state import Action, State


class PreferencePair(BaseModel):
    """One DPO example: at the first step where two paths for the same task differ, which action is better.

    The prompt is built from `state` exactly as for SFT (system prompt with the available tools, user prompt with the
    task and the history so far); chosen and rejected are the two candidate next actions.
    """
    task_id: str
    template: str
    kind: Literal["cost", "correctness"] = Field(
        description="cost: both paths succeed and chosen is cheaper. correctness: the rejected path fails."
    )
    state: State = Field(description="The state both paths share right before they diverge (task, tools, history).")
    chosen: Action
    rejected: Action
    chosen_path_cost: float = Field(description="Total cost of the full path the chosen action belongs to.")
    rejected_path_cost: float = Field(description="Total cost of the full path the rejected action belongs to.")


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    preference_pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preference_pairs.append(PreferencePair.model_validate_json(line))
    return preference_pairs