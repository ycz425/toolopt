import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.agents.agent import Agent
from app.data.labeled_task import LabeledTask
from app.environment.state import State
from app.evaluation.metrics import argument_accuracy, num_tool_calls, task_success, tool_selection_accuracy, total_cost, total_latency


class TaskOutcome(BaseModel):
    """Stage 1 output: a task's final state, or None if the model never produced a schema-valid action."""
    task_id: str
    state: State | None


class BenchmarkResult(BaseModel):
    task_id: str
    success: bool
    tool_selection_accuracy: float
    argument_accuracy: float
    num_tool_calls: int
    total_latency: float
    total_cost: float


def _completed_task_ids(path: Path) -> set[str]:
    """task_ids already recorded in a states or results file (every line of both has a task_id)."""
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def _append(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


async def run_benchmark(agent: Agent, labeled_tasks: list[LabeledTask], states_path: Path) -> None:
    """Stage 1: runs the agent on each task and appends its final state to states_path as soon as it finishes.

    Tasks already in states_path are skipped, so an interrupted run resumes where it stopped.
    """
    done = _completed_task_ids(states_path)
    remaining = [t for t in labeled_tasks if t.task.task_id not in done]
    print(f"{len(done)} tasks already run, {len(remaining)} remaining")
    for i, labeled_task in enumerate(remaining, start=1):
        try:
            state = await agent.run(labeled_task.task)
            status = "completed"
        except ValidationError:
            # The policy exhausted its retries without producing a schema-valid action -- an expected
            # failure mode for an imperfect model, recorded and scored as a failed task.
            state = None
            status = "never produced a valid action"
        _append(states_path, TaskOutcome(task_id=labeled_task.task.task_id, state=state))
        print(f"{datetime.now()}     [{i}/{len(remaining)}] {labeled_task.task.task_id}: {status}")


def load_outcomes(states_path: Path) -> dict[str, TaskOutcome]:
    with states_path.open() as f:
        outcomes = [TaskOutcome.model_validate_json(line) for line in f if line.strip()]
    return {o.task_id: o for o in outcomes}


def load_results(results_path: Path) -> list[BenchmarkResult]:
    with results_path.open() as f:
        return [BenchmarkResult.model_validate_json(line) for line in f if line.strip()]


async def evaluate_benchmark(labeled_tasks: list[LabeledTask], states_path: Path, results_path: Path) -> list[BenchmarkResult]:
    """Stage 2: scores the final states saved by run_benchmark, appending each result to results_path.

    Tasks already in results_path are skipped, so an evaluation stopped by the judge's rate limit resumes
    where it stopped. Tasks with no saved state yet (stage 1 unfinished) are left out. Returns every result
    in results_path for the given tasks.
    """
    outcomes = load_outcomes(states_path)
    done = _completed_task_ids(results_path)
    missing = [t.task.task_id for t in labeled_tasks if t.task.task_id not in outcomes]
    if missing:
        print(f"WARNING: {len(missing)} tasks have no saved state yet and are not evaluated; finish stage 1 first")

    remaining = [t for t in labeled_tasks if t.task.task_id in outcomes and t.task.task_id not in done]
    print(f"{len(done)} tasks already evaluated, {len(remaining)} remaining")
    for i, labeled_task in enumerate(remaining, start=1):
        state = outcomes[labeled_task.task.task_id].state
        if state is None:
            result = BenchmarkResult(
                task_id=labeled_task.task.task_id,
                success=False,
                tool_selection_accuracy=0.0,
                argument_accuracy=0.0,
                num_tool_calls=0,
                total_latency=0.0,
                total_cost=0.0,
            )
        else:
            result = BenchmarkResult(
                task_id=labeled_task.task.task_id,
                success=await task_success(state, labeled_task.expected_answer),
                tool_selection_accuracy=tool_selection_accuracy(state, labeled_task.expected_actions),
                argument_accuracy=argument_accuracy(state, labeled_task.expected_actions),
                num_tool_calls=num_tool_calls(state),
                total_latency=total_latency(state),
                total_cost=total_cost(state),
            )
        _append(results_path, result)
        print(f"{datetime.now()}     [{i}/{len(remaining)}] {labeled_task.task.task_id}: {'success' if result.success else 'failed'}")

    wanted = {t.task.task_id for t in labeled_tasks}
    return [r for r in load_results(results_path) if r.task_id in wanted]


def summarize(results: list[BenchmarkResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "num_tasks": n,
        "success_rate": sum(r.success for r in results) / n,
        "avg_tool_selection_accuracy": sum(r.tool_selection_accuracy for r in results) / n,
        "avg_argument_accuracy": sum(r.argument_accuracy for r in results) / n,
        "avg_num_tool_calls": sum(r.num_tool_calls for r in results) / n,
        "avg_latency": sum(r.total_latency for r in results) / n,
        "avg_cost": sum(r.total_cost for r in results) / n,
    }
