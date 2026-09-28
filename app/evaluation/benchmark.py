from datetime import datetime

from pydantic import BaseModel, ValidationError

from app.agents.agent import Agent
from app.data.labeled_task import LabeledTask
from app.evaluation.metrics import argument_accuracy, num_tool_calls, task_success, tool_selection_accuracy, total_cost, total_latency


class BenchmarkResult(BaseModel):
    task_id: str
    success: bool
    tool_selection_accuracy: float
    argument_accuracy: float
    num_tool_calls: int
    total_latency: float
    total_cost: float


async def run_benchmark(agent: Agent, labeled_tasks: list[LabeledTask]) -> list[BenchmarkResult]:
    results = []
    for i, labeled_task in enumerate(labeled_tasks, start=1):
        try:
            state = await agent.run(labeled_task.task)
            result = BenchmarkResult(
                task_id=labeled_task.task.task_id,
                success=await task_success(state, labeled_task.expected_answer),
                tool_selection_accuracy=tool_selection_accuracy(state, labeled_task.expected_actions),
                argument_accuracy=argument_accuracy(state, labeled_task.expected_actions),
                num_tool_calls=num_tool_calls(state),
                total_latency=total_latency(state),
                total_cost=total_cost(state),
            )
            status = "success" if result.success else "failed"
        except ValidationError:
            # The policy exhausted its retries without producing a schema-valid action -- an expected
            # failure mode for an imperfect model, recorded as a failed task instead of ending the run.
            result = BenchmarkResult(
                task_id=labeled_task.task.task_id,
                success=False,
                tool_selection_accuracy=0.0,
                argument_accuracy=0.0,
                num_tool_calls=0,
                total_latency=0.0,
                total_cost=0.0,
            )
            status = "failed (never produced a valid action)"
        results.append(result)
        print(f"{datetime.now()}     [{i}/{len(labeled_tasks)}] {labeled_task.task.task_id}: {status}")
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