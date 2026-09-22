import asyncio
import json
from pathlib import Path

from app.agents.agent import Agent
from app.agents.gemini_policy import GeminiPolicy
from app.data.generator import generate_trajectory
from app.data.task_bank import build_task_bank
from app.data.trajectory import Trajectory
from app.environment.environment import Environment
from app.evaluation.benchmark import BenchmarkCase
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry

OUTPUT_PATH = Path("datasets/trajectories/trajectories.jsonl")
BACKOFF_SECONDS = 60
PROGRESS_PRINT_EVERY = 25


def load_completed_ids() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    ids = set()
    with OUTPUT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["task"]["task_id"])
    return ids


def append_trajectory(trajectory: Trajectory) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a") as f:
        f.write(trajectory.model_dump_json() + "\n")


def _is_rate_limit_error(e: Exception) -> bool:
    # Prefer a structured status code if the SDK's exception exposes one; fall back to checking
    # for a literal "429" in the stringified error otherwise.
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code == 429:
        return True
    return "429" in str(e)


MAX_CONSECUTIVE_BACKOFFS = 5


async def generate_one(agent: Agent, case: BenchmarkCase) -> Trajectory | None:
    """Returns a Trajectory once the case completes. Returns None once MAX_CONSECUTIVE_BACKOFFS
    backoffs in a row have failed to clear -- Gemini's error responses don't reliably distinguish
    RPM/TPM from RPD, so this uses repeated-failure count as a proxy instead of parsing the error
    text: a per-minute/per-token limit should clear within a backoff or two, so failing several
    times in a row (several minutes) points to something longer-lived, i.e. the daily quota."""
    backoffs = 0
    while True:
        try:
            return await generate_trajectory(agent, case)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            backoffs += 1
            if backoffs > MAX_CONSECUTIVE_BACKOFFS:
                print(f"Rate limited {backoffs} times in a row on {case.task.task_id} -- assuming daily quota: {e}")
                return None
            print(f"Rate limited on {case.task.task_id} (attempt {backoffs}/{MAX_CONSECUTIVE_BACKOFFS}) -- backing off {BACKOFF_SECONDS}s: {e}")
            await asyncio.sleep(BACKOFF_SECONDS)


async def main():
    registry = build_registry()
    executor = ToolExecutor()
    environment = Environment(registry, executor)
    agent = Agent(environment, GeminiPolicy())

    all_cases = build_task_bank()
    done = load_completed_ids()
    remaining = [c for c in all_cases if c.task.task_id not in done]

    print(f"{len(done)} already done, {len(remaining)} remaining out of {len(all_cases)} total")

    completed_this_run = 0
    for case in remaining:
        trajectory = await generate_one(agent, case)
        if trajectory is None:
            print(f"Stopping for today after {completed_this_run} cases this run. Rerun this script "
                  "tomorrow (or once quota resets) to resume where it left off.")
            return
        append_trajectory(trajectory)
        completed_this_run += 1
        if completed_this_run % PROGRESS_PRINT_EVERY == 0:
            print(f"{completed_this_run}/{len(remaining)} done this run")

    print(f"All {len(all_cases)} cases complete.")


if __name__ == "__main__":
    asyncio.run(main())
