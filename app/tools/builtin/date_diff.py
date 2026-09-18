from datetime import date

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolMetadata, ToolParameter


class DateDiffResult(BaseModel):
    days: int = Field(description="Number of days between the two dates (positive if date2 is after date1).")


class DateDiffTool(Tool):
    metadata = ToolMetadata(
        name="date_diff",
        description="Compute the number of days between two dates.",
        parameters=[
            ToolParameter(name="date1", type="string", description="First date, in YYYY-MM-DD format."),
            ToolParameter(name="date2", type="string", description="Second date, in YYYY-MM-DD format."),
        ],
        returns=DateDiffResult.__name__,
    )

    def execute(self, date1: str, date2: str) -> DateDiffResult:
        d1, d2 = date.fromisoformat(date1), date.fromisoformat(date2)
        return DateDiffResult(days=(d2 - d1).days)

    def cost(self, date1: str, date2: str) -> float:
        return 0.01
