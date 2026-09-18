from app.agents.policy import Policy
from app.environment.environment import Environment
from app.environment.task import Task
from datetime import datetime


class Agent:
    def __init__(self, environment: Environment, policy: Policy, verbose=False):
        self.environment = environment
        self.policy = policy
        self.verbose = verbose

    async def run(self, task: Task, temperature: float = 0):
        if self.verbose:
            print()
        state = self.environment.reset(task)
        while not self.environment.is_done():
            action = await self.policy.select_action(task, state.available_tools, state, temperature=temperature)
            state = self.environment.step(action)
            if self.verbose:
                print(f'{datetime.now()}     Selected tool {action.tool_name} with args {action.args}')
        return state