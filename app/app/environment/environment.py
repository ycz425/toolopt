from app.environment.state import State, Action, Step
from app.environment.task import Task
from app.tools.registry import ToolRegistry
from app.tools.executor import ToolExecutor

class Environment:
    def __init__(self, tool_registry: ToolRegistry, tool_executor: ToolExecutor, max_steps: int = 10):
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def reset(self, task: Task) -> State:
        for tool in self.tool_registry.all_tools():
            tool.reset()
        self.state = State(task=task, available_tools=self.tool_registry.list())
        return self.state

    def step(self, action: Action) -> State:
        tool = self.tool_registry.get(action.tool_name)
        result = self.tool_executor.execute(tool, **action.args)
        step = Step(action=action, result=result)
        self.state = self.state.model_copy(update={'history': [*self.state.history, step]})
        return self.state

    def get_state(self) -> State:
        return self.state

    def is_done(self) -> bool:
        if self.state.history and self.state.history[-1].action.tool_name == 'finish':
            return True
        return len(self.state.history) >= self.max_steps