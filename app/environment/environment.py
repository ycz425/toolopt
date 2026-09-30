from app.environment.state import State, Action, Step
from app.environment.task import Task
from app.tools.registry import ToolRegistry
from app.tools.base import ToolMetadata
from app.tools.executor import ToolExecutionResult, ToolExecutor

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
        if action.tool_name in self.tool_registry.tools:
            result = self.tool_executor.execute(self.tool_registry.get(action.tool_name), **action.args)
        else:
            # A model can name a tool that doesn't exist. Record it as a failed call, like a tool that
            # raised, so the agent sees the error in its history and can recover instead of crashing.
            result = ToolExecutionResult(
                metadata=ToolMetadata(name=action.tool_name, description="Unknown tool.", parameters=[], returns="None"),
                args=action.args,
                latency=0.0,
                success=False,
                cost=None,
                error=f"tool '{action.tool_name}' does not exist",
                output=None,
            )
        step = Step(action=action, result=result)
        self.state = self.state.model_copy(update={'history': [*self.state.history, step]})
        return self.state

    def get_state(self) -> State:
        return self.state

    def is_done(self) -> bool:
        if self.state.history and self.state.history[-1].action.tool_name == 'finish':
            return True
        return len(self.state.history) >= self.max_steps