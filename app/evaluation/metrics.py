from app.agents.judge import Judge
from app.environment.state import Action, State

_judge: Judge | None = None


def _get_judge() -> Judge:
    global _judge
    if _judge is None:
        _judge = Judge()
    return _judge


def _jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def _value_matches(actual, expected, tol: float = 1e-6) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) < tol
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip().lower() == expected.strip().lower()
    return actual == expected


async def task_success(state: State, expected_answer: str) -> bool:
    if not state.history or state.history[-1].action.tool_name != "finish":
        return False

    actual_answer = state.history[-1].result.output["answer"]
    result = await _get_judge().judge(state.task, expected_answer, actual_answer)
    return result.correct


def tool_selection_accuracy(state: State, expected_actions: list[Action]) -> float:
    s1 = {step.action.tool_name for step in state.history}
    s2 = {action.tool_name for action in expected_actions}
    return _jaccard(s1, s2)


def argument_accuracy(state: State, expected_actions: list[Action]) -> float:
    expected_args = {action.tool_name: action.args for action in expected_actions}

    calls = {step.action.tool_name: step.action.args for step in state.history}
    total_fields = sum(len(fields) for fields in expected_args.values())
    if total_fields == 0:
        return 1.0

    correct_fields = 0
    for tool, fields in expected_args.items():
        actual = calls.get(tool, {})
        for key, expected_value in fields.items():
            if key in actual and _value_matches(actual[key], expected_value):
                correct_fields += 1
    return correct_fields / total_fields

def num_tool_calls(state: State) -> int:
    return len(state.history)

def total_latency(state: State) -> float:
    return sum(step.result.latency for step in state.history)

def total_cost(state: State) -> float:
    # step.result.cost is None for failed calls (ToolExecutor never computes tool.cost() on
    # failure), so failed steps contribute 0 rather than breaking the sum -- consistent with
    # the executor's existing behavior of not charging for calls that raised.
    return sum(step.result.cost or 0.0 for step in state.history)
