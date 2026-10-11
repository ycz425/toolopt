import statistics as stats_module
from typing import Literal

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter, ToolPricing

StatMode = Literal["mean", "median", "stdev"]


class StatsResult(BaseModel):
    value: float = Field(description="The computed statistic.")


class StatsTool(Tool):
    metadata = ToolMetadata(
        name="statistics",
        description="Compute the mean, median, or standard deviation of a list of numbers.",
        pricing=ToolPricing(base=0.01),
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


class DescribeNumbersResult(BaseModel):
    mean: float = Field(description="The mean of the numbers.")
    median: float = Field(description="The median of the numbers.")
    stdev: float = Field(description="The sample standard deviation of the numbers.")
    min: float = Field(description="The smallest number.")
    max: float = Field(description="The largest number.")


class DescribeNumbersTool(Tool):
    metadata = ToolMetadata(
        name="describe_numbers",
        description="Compute the mean, median, standard deviation, min, and max of a list of numbers in one call.",
        pricing=ToolPricing(base=0.015),
        parameters=[
            ToolParameter(name="numbers", type="array", description="List of at least two numbers."),
        ],
        returns=DescribeNumbersResult.__name__,
    )

    def execute(self, numbers: list[float]) -> DescribeNumbersResult:
        if len(numbers) < 2:
            raise ValueError("describe_numbers needs at least two numbers")
        return DescribeNumbersResult(
            mean=stats_module.mean(numbers),
            median=stats_module.median(numbers),
            stdev=stats_module.stdev(numbers),
            min=min(numbers),
            max=max(numbers),
        )
