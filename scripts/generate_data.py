import asyncio
import json
from pathlib import Path

from app.agents.agent import Agent
from app.agents.policy import Policy
from app.data.generator import generate_trajectory
from app.data.task_bank import build_task_bank
from app.data.trajectory import Trajectory
from app.environment.environment import Environment
from app.evaluation.benchmark import BenchmarkCase
from app.tools.builtin.add_days import AddDaysTool
from app.tools.builtin.aggregate_records import AggregateRecordsTool
from app.tools.builtin.calculator import CalculatorTool
from app.tools.builtin.calendar import CalendarCreateTool, CalendarListTool, CalendarStore
from app.tools.builtin.compound_interest import CompoundInterestTool
from app.tools.builtin.converter import ConverterTool
from app.tools.builtin.date_diff import DateDiffTool
from app.tools.builtin.distance import DistanceTool
from app.tools.builtin.email import SendEmailTool
from app.tools.builtin.encoding import Base64DecodeTool, Base64EncodeTool
from app.tools.builtin.filter_records import FilterRecordsTool
from app.tools.builtin.files import FileReadTool, FileWriteTool, VirtualFileSystem
from app.tools.builtin.percentile import PercentileTool
from app.tools.builtin.prime_checker import PrimeCheckerTool
from app.tools.builtin.roman_numeral import RomanNumeralTool
from app.tools.builtin.scientific_calculator import ScientificCalculatorTool
from app.tools.builtin.stats import StatsTool
from app.tools.builtin.stock_price import StockPriceTool
from app.tools.builtin.temperature import TemperatureConverterTool
from app.tools.builtin.weather import WeatherTool
from app.tools.builtin.weather_forecast import WeatherForecastTool
from app.tools.builtin.word_count import WordCountTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

OUTPUT_PATH = Path("datasets/trajectories/trajectories.jsonl")
BACKOFF_SECONDS = 60
PROGRESS_PRINT_EVERY = 25


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()  # FinishTool is auto-registered by ToolRegistry itself
    registry.register(CalculatorTool())
    registry.register(ConverterTool())
    registry.register(WeatherTool())
    registry.register(StockPriceTool())
    registry.register(WordCountTool())
    registry.register(PrimeCheckerTool())
    registry.register(RomanNumeralTool())
    registry.register(StatsTool())
    registry.register(DateDiffTool())
    registry.register(TemperatureConverterTool())
    registry.register(Base64EncodeTool())
    registry.register(Base64DecodeTool())
    registry.register(DistanceTool())
    registry.register(CompoundInterestTool())
    registry.register(FilterRecordsTool())
    registry.register(WeatherForecastTool())
    registry.register(ScientificCalculatorTool())
    registry.register(PercentileTool())
    registry.register(AggregateRecordsTool())
    registry.register(AddDaysTool())

    calendar_store = CalendarStore()
    registry.register(CalendarListTool(calendar_store))
    registry.register(CalendarCreateTool(calendar_store))

    fs = VirtualFileSystem()
    registry.register(FileReadTool(fs))
    registry.register(FileWriteTool(fs))

    registry.register(SendEmailTool())

    return registry


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
    agent = Agent(environment, Policy())

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
