from pydantic import BaseModel

from app.agents.agent import Agent
from app.environment.state import Action
from app.environment.task import Task
from app.evaluation.metrics import argument_accuracy, num_tool_calls, task_success, tool_selection_accuracy, total_cost, total_latency


class BenchmarkCase(BaseModel):
    task: Task
    expected_answer: str
    expected_actions: list[Action]


class BenchmarkResult(BaseModel):
    task_id: str
    success: bool
    tool_selection_accuracy: float
    argument_accuracy: float
    num_tool_calls: int
    total_latency: float
    total_cost: float


async def run_benchmark(agent: Agent, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
    results = []
    for case in cases:
        state = await agent.run(case.task)
        results.append(BenchmarkResult(
            task_id=case.task.task_id,
            success=await task_success(state, case.expected_answer),
            tool_selection_accuracy=tool_selection_accuracy(state, case.expected_actions),
            argument_accuracy=argument_accuracy(state, case.expected_actions),
            num_tool_calls=num_tool_calls(state),
            total_latency=total_latency(state),
            total_cost=total_cost(state),
        ))
    return results


def summarize(results: list[BenchmarkResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "success_rate": sum(r.success for r in results) / n,
        "avg_tool_selection_accuracy": sum(r.tool_selection_accuracy for r in results) / n,
        "avg_argument_accuracy": sum(r.argument_accuracy for r in results) / n,
        "avg_num_tool_calls": sum(r.num_tool_calls for r in results) / n,
        "avg_latency": sum(r.total_latency for r in results) / n,
        "avg_cost": sum(r.total_cost for r in results) / n,
    }