from app.data.trajectory import Trajectory
from app.environment.environment import Environment
from app.data.labeled_task import LabeledTask


def generate_trajectories(environment: Environment, labeled_tasks: list[LabeledTask]) -> list[Trajectory]:
    trajectories = []
    for labeled_task in labeled_tasks:
        state = environment.reset(labeled_task.task)
        for action in labeled_task.expected_actions:
            state = environment.step(action)  # runs the tool locally
        success = all(s.result.success for s in state.history) and state.history[-1].action.tool_name == "finish"
        trajectory = Trajectory.from_state(state, success)
        trajectories.append(trajectory)
    return trajectories
