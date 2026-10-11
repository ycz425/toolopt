from app.data.trajectory import Trajectory
from app.environment.environment import Environment
from app.environment.state import Action
from app.environment.task import Task


def run_path(environment: Environment, task: Task, actions: list[Action]) -> Trajectory:
    """Runs a scripted sequence of actions through the environment (executing each tool locally).

    The trajectory succeeds if every call succeeded and the last action was finish.
    """
    state = environment.reset(task)
    for action in actions:
        state = environment.step(action)
    success = all(s.result.success for s in state.history) and state.history[-1].action.tool_name == "finish"
    return Trajectory.from_state(state, success)
