from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class PercentileResult(BaseModel):
    value: float = Field(description="The value at the requested percentile.")


def compute_percentile(numbers: list[float], p: float) -> float:
    if not numbers:
        raise ValueError("numbers must not be empty")
    if not (0 <= p <= 100):
        raise ValueError("p must be between 0 and 100")
    sorted_nums = sorted(numbers)
    k = (len(sorted_nums) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_nums) - 1)
    if f == c:
        return sorted_nums[f]
    return sorted_nums[f] + (sorted_nums[c] - sorted_nums[f]) * (k - f)


class PercentileTool(Tool):
    metadata = ToolMetadata(
        name="percentile",
        description="Compute the Nth percentile of a list of numbers.",
        parameters=[
            ToolParameter(name="numbers", type="array", description="List of numbers."),
            ToolParameter(name="p", type="number", description="Percentile to compute, e.g. 90 for the 90th percentile."),
        ],
        returns=PercentileResult.__name__,
    )

    def execute(self, numbers: list[float], p: float) -> PercentileResult:
        return PercentileResult(value=round(compute_percentile(numbers, p), 2))

    def cost(self, numbers: list[float], p: float) -> float:
        return 0.01
