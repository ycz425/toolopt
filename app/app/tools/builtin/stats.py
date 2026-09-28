import statistics as stats_module
from typing import Literal

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter

StatMode = Literal["mean", "median", "stdev"]


class StatsResult(BaseModel):
    value: float = Field(description="The computed statistic.")


class StatsTool(Tool):
    metadata = ToolMetadata(
        name="statistics",
        description="Compute the mean, median, or standard deviation of a list of numbers.",
        parameters=[
            ToolParameter(name="numbers", type="array", description="List of numbers."),
            ToolParameter(name="stat", type="string", description="One of 'mean', 'median', 'stdev'."),
        ],
        returns=StatsResult.__name__,
    )

    def execute(self, numbers: list[float], stat: StatMode) -> StatsResult:
        if not numbers:
            raise ValueError("numbers must not be empty")
        if stat == "mean":
            return StatsResult(value=stats_module.mean(numbers))
        if stat == "median":
            return StatsResult(value=stats_module.median(numbers))
        if stat == "stdev":
            return StatsResult(value=stats_module.stdev(numbers))
        raise ValueError(f"unknown stat '{stat}'")

    def cost(self, numbers: list[float], stat: StatMode) -> float:
        return 0.01
