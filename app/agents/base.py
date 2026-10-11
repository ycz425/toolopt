from abc import ABC, abstractmethod

from app.environment.state import Action, State
from app.environment.task import Task
from app.tools.base import ToolMetadata


class Policy(ABC):
    @abstractmethod
    async def select_action(
        self, task: Task, tools: list[ToolMetadata], state: State, temperature: float = 0
    ) -> Action:
        ...
