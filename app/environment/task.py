from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(description="Stable unique identifier for the task, e.g. 'weather_001'. Used to key results, trajectories, and dataset splits.")
    description: str = Field(description="Natural-language statement of what the agent is trying to accomplish.")